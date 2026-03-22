"""
SAM3 Two-Mouse Tracking Node for ComfyUI

Accumulates point prompts from SAM3PointCollector and builds a video state
for SAM3Propagate.

Workflow:
  SAM3PointCollector ──► SAM3TwoMouseTracking ──► SAM3Propagate ──► SAM3VideoOutput

Usage:
  1. Connect video_frames to SAM3PointCollector (upgraded).
  2. Set frame_idx / animal_id in SAM3PointCollector, click on the animal.
  3. Queue (Ctrl+Enter) → the prompt is appended to SAM3TwoMouseTracking's list.
  4. Repeat steps 2-3 for each animal / frame you want to annotate.
  5. Click "Clear List" in SAM3TwoMouseTracking to start over.
"""
import hashlib
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch

try:
    import server
    from aiohttp import web
    _SERVER_AVAILABLE = True
except Exception:
    server = None
    _SERVER_AVAILABLE = False

log = logging.getLogger("sam3")

from .video_state import (
    SAM3VideoState,
    VideoPrompt,
    VideoConfig,
    create_video_state,
    create_video_state_from_file,
)
from .utils import print_mem


class SAM3TwoMouseTracking:
    """
    Accumulates SAM3_SINGLE_PROMPT entries and builds SAM3_VIDEO_STATE.

    Each time the upstream SAM3PointCollector runs with new points, this node
    appends (frame_idx, animal_id, points) to an internal prompt list and
    rebuilds the video state.

    Use the "Clear List" button to reset the accumulated prompts.
    """

    # Cache for video states keyed by (video_hash, prompt_store_hash)
    _state_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_store": ("STRING", {
                    "multiline": False,
                    "default": "[]",
                    "tooltip": "JSON array of accumulated prompts (managed automatically).",
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
            },
            "optional": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch [N, H, W, C]."
                }),
                "video": ("VIDEO", {
                    "tooltip": "Video from Load Video node. Takes priority over video_frames."
                }),
                "new_prompt": ("SAM3_SINGLE_PROMPT", {
                    "tooltip": "Single prompt from SAM3PointCollector. Appended each run."
                }),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "segment"
    CATEGORY = "SAM3/video"
    OUTPUT_NODE = True  # Run even without downstream connections

    @classmethod
    def IS_CHANGED(cls, prompt_store, score_threshold,
                   video_frames=None, video=None, new_prompt=None):
        # Always re-run so new prompts are appended on every queue.
        return float("nan")

    def segment(self, prompt_store, score_threshold,
                video_frames=None, video=None, new_prompt=None):

        if video is None and video_frames is None:
            raise ValueError("Either video or video_frames must be provided.")

        # ── Load accumulated prompts ──────────────────────────────────────
        try:
            parsed = json.loads(prompt_store) if str(prompt_store).strip() else []
            prompts = parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            prompts = []

        # ── Append new prompt if provided ─────────────────────────────────
        if new_prompt and new_prompt.get("points"):
            prompts.append(new_prompt)
            log.info(f"Appended prompt: frame={new_prompt['frame_idx']} "
                     f"obj={new_prompt['obj_id']} "
                     f"pts={len(new_prompt['points'])}")
        else:
            log.info("No new prompt — rebuilding from existing list.")

        updated_store = json.dumps(prompts)
        log.info(f"Total accumulated prompts: {len(prompts)}")

        # ── Video hash for state cache ────────────────────────────────────
        vh = hashlib.md5()
        if video is not None:
            try:
                source = video.get_stream_source()
                if isinstance(source, str):
                    vh.update(source.encode())
                    try:
                        vh.update(str(os.path.getmtime(source)).encode())
                    except OSError:
                        pass
                else:
                    vh.update(str(video.get_frame_count()).encode())
            except Exception:
                vh.update(str(id(video)).encode())
        else:
            vh.update(str(video_frames.shape).encode())
        video_hash = vh.hexdigest()

        ph = hashlib.md5()
        ph.update(updated_store.encode())
        ph.update(str(score_threshold).encode())
        cache_key = video_hash + ph.hexdigest()

        if cache_key in SAM3TwoMouseTracking._state_cache:
            log.info(f"STATE CACHE HIT key={cache_key[:8]}")
            video_state = SAM3TwoMouseTracking._state_cache[cache_key]
            return {
                "ui": {
                    "prompt_store":  [updated_store],
                    "prompt_count": [str(len(prompts))],
                },
                "result": (video_state,),
            }

        log.info(f"STATE CACHE MISS key={cache_key[:8]} — building video state")
        print_mem("Before SAM3TwoMouseTracking")

        # ── Create base video state ───────────────────────────────────────
        config = VideoConfig(score_threshold_detection=score_threshold)

        if video is not None:
            try:
                source = video.get_stream_source()
                if isinstance(source, str) and os.path.isfile(source):
                    video_state = create_video_state_from_file(source, config=config)
                else:
                    components = video.get_components()
                    video_state = create_video_state(components.images, config=config)
            except Exception:
                components = video.get_components()
                video_state = create_video_state(components.images, config=config)
        else:
            video_state = create_video_state(video_frames, config=config)

        log.info(f"Session {video_state.session_uuid[:8]}: "
                 f"{video_state.num_frames} frames, "
                 f"{video_state.width}x{video_state.height}")

        # ── Apply all accumulated prompts ─────────────────────────────────
        for entry in prompts:
            frame_idx = entry["frame_idx"]
            obj_id    = entry["obj_id"]
            pts_px    = entry["points"]
            labels    = entry["labels"]
            img_w     = entry["img_width"]
            img_h     = entry["img_height"]

            pts_norm = [[x / img_w, y / img_h] for x, y in pts_px]
            prompt = VideoPrompt.create_point(frame_idx, obj_id, pts_norm, labels)
            video_state = video_state.with_prompt(prompt)

            pos = sum(l == 1 for l in labels)
            neg = sum(l == 0 for l in labels)
            log.info(f"  frame={frame_idx} obj={obj_id} {pos} pos {neg} neg")

        print_mem("After SAM3TwoMouseTracking")
        SAM3TwoMouseTracking._state_cache[cache_key] = video_state

        return {
            "ui": {
                "prompt_store":  [updated_store],
                "prompt_count": [str(len(prompts))],
            },
            "result": (video_state,),
        }


# =============================================================================
# SAM3FrameCorrector
# =============================================================================

class SAM3FrameCorrector:
    """
    Applies frame-level mask corrections to existing propagation results.

    Instead of re-running full propagation, this node calls add_prompt on
    specific frames and UNIONs the returned mask into the existing base mask.
    Other objects always take priority (no overlap).

    Workflow:
      SAM3Propagate ──► SAM3FrameCorrector ──► SAM3VideoOutput
                              ↑
                    SAM3PointCollector (correction prompts)

    Usage:
      1. Run SAM3Propagate with your initial prompts.
      2. Connect SAM3Propagate's masks + video_state outputs here.
      3. Use a separate SAM3PointCollector to add correction points.
      4. Queue → corrected mask appears only for the specified frame.
      5. Click "Clear Corrections" to start over.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model (from LoadSAM3Model).",
                }),
                "masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Base masks from SAM3Propagate.",
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state from SAM3Propagate (used for session access).",
                }),
                "correction_store": ("STRING", {
                    "multiline": False,
                    "default": "[]",
                    "tooltip": "JSON array of accumulated corrections (managed automatically).",
                }),
            },
            "optional": {
                "new_correction": ("SAM3_SINGLE_PROMPT", {
                    "tooltip": "Single correction prompt from SAM3PointCollector.",
                }),
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames (from Load Video). Used to rebuild the session if temp files were deleted.",
                }),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS",)
    RETURN_NAMES = ("corrected_masks",)
    FUNCTION = "correct"
    CATEGORY = "SAM3/video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, sam3_model, masks, video_state,
                   correction_store, new_correction=None, video_frames=None):
        return float("nan")

    def _segment_frame(self, sam3_model, frame_tensor, obj_id, pts_norm, labels):
        """
        Segment a single frame using the SAM3 image predictor (predict_inst).
        Uses point prompts directly on one frame — no propagation or session needed.

        Args:
            frame_tensor: [H, W, C] float32 ComfyUI image tensor (0–1)
            obj_id: object id (1-indexed, unused for image predictor but kept for API compat)
            pts_norm: normalized point coords [[x, y], ...]  (0–1)
            labels: point labels [1=positive, 0=negative, ...]

        Returns:
            2-D float tensor [H, W] (0/1) on CPU, or None on failure
        """
        try:
            import comfy.model_management
            from PIL import Image as PILImage

            comfy.model_management.load_models_gpu([sam3_model])

            frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = PILImage.fromarray(frame_np)

            processor = sam3_model.processor
            model = processor.model
            if hasattr(processor, "sync_device_with_model"):
                processor.sync_device_with_model()

            state = processor.set_image(pil_img)

            pts_np    = np.array(pts_norm, dtype=np.float32)
            labels_np = np.array(labels,   dtype=np.int32)

            masks_np, scores_np, _ = model.predict_inst(
                state,
                point_coords=pts_np,
                point_labels=labels_np,
                box=None,
                mask_input=None,
                multimask_output=True,
                normalize_coords=False,  # pts_np is already 0-1 normalized
            )

            best_idx = int(np.argmax(scores_np))
            mask = torch.from_numpy(masks_np[best_idx]).float()   # [H, W]
            log.info(f"_segment_frame: mask shape={mask.shape} score={scores_np[best_idx]:.3f}")
            return mask

        except Exception as e:
            log.warning(f"_segment_frame failed: {e}")
            return None

    def correct(self, sam3_model, masks, video_state,
                correction_store, new_correction=None, video_frames=None):

        # ── Load accumulated corrections ──────────────────────────────────
        try:
            parsed = json.loads(correction_store) if str(correction_store).strip() else []
            corrections = parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            corrections = []

        # ── Append new correction if provided ─────────────────────────────
        if new_correction and new_correction.get("points"):
            corrections.append(new_correction)
            log.info(f"Appended correction: frame={new_correction['frame_idx']} "
                     f"obj={new_correction['obj_id']} "
                     f"pts={len(new_correction['points'])}")

        updated_store = json.dumps(corrections)
        log.info(f"Total corrections: {len(corrections)}")

        if not corrections:
            return {
                "ui": {
                    "correction_store": [updated_store],
                    "correction_count": ["0"],
                },
                "result": (masks,),
            }

        if video_frames is None:
            log.warning("video_frames not connected; corrections require the video_frames input")
            return {
                "ui": {
                    "correction_store": [updated_store],
                    "correction_count": [str(len(corrections))],
                },
                "result": (masks,),
            }

        # ── Work on a shallow copy of masks dict ──────────────────────────
        corrected = dict(masks)

        # ── Apply each correction via single-frame mini-session ───────────
        for entry in corrections:
            frame_idx = entry["frame_idx"]
            obj_id    = entry["obj_id"]
            pts_px    = entry["points"]
            labels    = entry["labels"]
            img_w     = entry["img_width"]
            img_h     = entry["img_height"]

            pts_norm = [[x / img_w, y / img_h] for x, y in pts_px]

            if frame_idx not in corrected:
                log.warning(f"Frame {frame_idx} not in base masks, skipping")
                continue

            if frame_idx >= video_frames.shape[0]:
                log.warning(f"frame_idx {frame_idx} out of video range ({video_frames.shape[0]}), skipping")
                continue

            log.info(f"Correcting frame={frame_idx} obj={obj_id} pts={len(pts_px)}")
            corr_raw = self._segment_frame(
                sam3_model, video_frames[frame_idx], obj_id, pts_norm, labels
            )

            if corr_raw is None:
                log.warning(f"No mask for frame {frame_idx} obj={obj_id}, skipping")
                continue

            if hasattr(corr_raw, "cpu"):
                corr_raw = corr_raw.cpu()
            if isinstance(corr_raw, np.ndarray):
                corr_raw = torch.from_numpy(corr_raw)
            while corr_raw.dim() > 3:
                corr_raw = corr_raw.squeeze(0)

            log.info(f"Correction mask shape at frame {frame_idx}: {corr_raw.shape}")

            # ── Get and normalise base mask [num_objs, H, W] ──────────────
            base_mask = corrected[frame_idx]
            if isinstance(base_mask, np.ndarray):
                base_mask = torch.from_numpy(base_mask)
            if base_mask.dim() == 4:
                base_mask = base_mask.squeeze(0)

            merged = base_mask.clone().float()
            if merged.numel() > 0 and merged.max() > 1.0:
                merged = merged / 255.0

            corr_f = corr_raw.float()
            if corr_f.numel() > 0 and corr_f.max() > 1.0:
                corr_f = corr_f / 255.0

            # obj_id is 1-indexed; tensor channel is 0-indexed
            obj_ch = obj_id - 1

            # Extract single-object slice from correction output
            if corr_f.dim() == 2:
                corr_single = corr_f
            elif corr_f.dim() == 3:
                if corr_f.shape[0] == 0:
                    log.warning(f"add_prompt returned 0 objects at frame {frame_idx}, skipping")
                    continue
                ch = obj_ch if obj_ch < corr_f.shape[0] else 0
                corr_single = corr_f[ch]
            else:
                log.warning(f"Unexpected corr_f shape: {corr_f.shape}, skipping")
                continue

            if obj_ch >= merged.shape[0]:
                log.warning(f"obj_ch {obj_ch} >= num_objects {merged.shape[0]}, skipping")
                continue

            corr_bin = (corr_single > 0.5).float()
            base_bin = (merged[obj_ch] > 0.5).float()

            has_pos = any(l == 1 for l in labels)
            has_neg = any(l == 0 for l in labels)

            if has_pos and not has_neg:
                # Positive only: UNION — add missing regions
                new_bin = (base_bin + corr_bin).clamp(0, 1)
                mode = "UNION"
            elif has_neg and not has_pos:
                # Negative only: INTERSECTION — remove false positives
                new_bin = base_bin * corr_bin
                mode = "INTERSECT"
            else:
                # Mixed: REPLACE with predictor result
                new_bin = corr_bin
                mode = "REPLACE"

            # Other objects take priority (corrected object yields)
            for other_ch in range(merged.shape[0]):
                if other_ch == obj_ch:
                    continue
                other_bin = (merged[other_ch] > 0.5).float()
                new_bin = new_bin * (1.0 - other_bin)

            merged[obj_ch] = new_bin
            corrected[frame_idx] = merged

            delta = int(new_bin.sum().item()) - int(base_bin.sum().item())
            log.info(f"Frame {frame_idx}: obj_id={obj_id} (ch={obj_ch}) [{mode}] "
                     f"{int(base_bin.sum())} → {int(new_bin.sum())} px "
                     f"({delta:+d} px)")

        return {
            "ui": {
                "correction_store": [updated_store],
                "correction_count": [str(len(corrections))],
            },
            "result": (corrected,),
        }


# =============================================================================
# SAM3SavePropagation / SAM3LoadPropagation
# =============================================================================

class SAM3SavePropagation:
    """
    Save SAM3Propagate outputs (masks + video_state) to a file.

    Workflow:
      SAM3Propagate ──► SAM3SavePropagation  (run once)
      SAM3LoadPropagation ──► SAM3FrameCorrector  (run repeatedly for corrections)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Masks from SAM3Propagate.",
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state from SAM3Propagate.",
                }),
                "filename": ("STRING", {
                    "default": "sam3_result",
                    "tooltip": "Filename (without extension). Saved under ComfyUI output/sam3_propagation/.",
                }),
            },
            "optional": {
                "scores": ("SAM3_VIDEO_SCORES", {
                    "tooltip": "Scores from SAM3Propagate (optional).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "save"
    CATEGORY = "SAM3/video"
    OUTPUT_NODE = True

    def save(self, masks, video_state, filename, scores=None):
        import folder_paths

        out_dir = os.path.join(folder_paths.get_output_directory(), "sam3_propagation")
        os.makedirs(out_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(out_dir, f"{filename}_{ts}.pt")

        data = {
            "masks": masks,
            "video_state_dict": video_state.to_dict(),
            "scores": scores if scores is not None else {},
        }
        torch.save(data, save_path)

        log.info(f"Saved propagation to {save_path} "
                 f"({len(masks)} frames, session={video_state.session_uuid[:8]})")
        print(f"[SAM3] Saved propagation → {save_path}")

        return {
            "ui": {"save_path": [save_path]},
            "result": (save_path,),
        }


class SAM3LoadPropagation:
    """
    Load SAM3Propagate outputs previously saved by SAM3SavePropagation.

    Enables a correction-only workflow that does not re-run SAM3Propagate:
      SAM3LoadPropagation ──► SAM3FrameCorrector ──► SAM3VideoOutput
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full path to the .pt file saved by SAM3SavePropagation.",
                }),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_STATE", "SAM3_VIDEO_SCORES")
    RETURN_NAMES = ("masks", "video_state", "scores")
    FUNCTION = "load"
    CATEGORY = "SAM3/video"

    @classmethod
    def IS_CHANGED(cls, save_path):
        try:
            return os.path.getmtime(save_path)
        except OSError:
            return float("nan")

    def load(self, save_path):
        if not os.path.exists(save_path):
            raise ValueError(f"[SAM3LoadPropagation] File not found: {save_path}")

        data = torch.load(save_path, map_location="cpu", weights_only=False)

        masks = data["masks"]
        video_state = SAM3VideoState.from_dict(data["video_state_dict"])
        scores = data.get("scores", {})

        log.info(f"Loaded propagation from {save_path}: "
                 f"{len(masks)} frames, session={video_state.session_uuid[:8]}")

        return (masks, video_state, scores)


# =============================================================================
# API route: print accumulated prompts to terminal
# =============================================================================

if _SERVER_AVAILABLE:
    @server.PromptServer.instance.routes.post("/sam3/list_prompts")
    async def _list_prompts_handler(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        prompt_store = body.get("prompt_store", "[]")
        try:
            parsed = json.loads(prompt_store) if str(prompt_store).strip() else []
            prompts = parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            prompts = []

        print("\n" + "=" * 60)
        print(f"[SAM3] Accumulated prompts ({len(prompts)} total):")
        for i, entry in enumerate(prompts):
            pos = sum(l == 1 for l in entry.get("labels", []))
            neg = sum(l == 0 for l in entry.get("labels", []))
            pts = entry.get("points", [])
            print(f"  [{i}] frame={entry.get('frame_idx')} "
                  f"obj={entry.get('obj_id')} "
                  f"pos={pos} neg={neg} "
                  f"pts={pts}")
        print("=" * 60 + "\n")

        return web.json_response({"count": len(prompts)})

    @server.PromptServer.instance.routes.post("/sam3/list_corrections")
    async def _list_corrections_handler(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        correction_store = body.get("correction_store", "[]")
        try:
            parsed = json.loads(correction_store) if str(correction_store).strip() else []
            corrections = parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            corrections = []

        print("\n" + "=" * 60)
        print(f"[SAM3] Accumulated corrections ({len(corrections)} total):")
        for i, entry in enumerate(corrections):
            pos = sum(l == 1 for l in entry.get("labels", []))
            neg = sum(l == 0 for l in entry.get("labels", []))
            pts = entry.get("points", [])
            print(f"  [{i}] frame={entry.get('frame_idx')} "
                  f"obj={entry.get('obj_id')} "
                  f"pos={pos} neg={neg} "
                  f"pts={pts}")
        print("=" * 60 + "\n")

        return web.json_response({"count": len(corrections)})

    @server.PromptServer.instance.routes.get("/sam3/list_propagations")
    async def _list_propagations_handler(request):
        import folder_paths
        out_dir = os.path.join(folder_paths.get_output_directory(), "sam3_propagation")
        files = []
        if os.path.isdir(out_dir):
            for fname in sorted(os.listdir(out_dir), reverse=True):
                if fname.endswith(".pt"):
                    full = os.path.join(out_dir, fname)
                    files.append({"path": full, "label": fname})
        return web.json_response(files)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM3TwoMouseTracking": SAM3TwoMouseTracking,
    "SAM3FrameCorrector": SAM3FrameCorrector,
    "SAM3SavePropagation": SAM3SavePropagation,
    "SAM3LoadPropagation": SAM3LoadPropagation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3TwoMouseTracking": "SAM3 Two-Mouse Tracking",
    "SAM3FrameCorrector": "SAM3 Frame Corrector",
    "SAM3SavePropagation": "SAM3 Save Propagation",
    "SAM3LoadPropagation": "SAM3 Load Propagation",
}
