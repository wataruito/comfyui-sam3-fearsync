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

    RETURN_TYPES = ("SAM3_VIDEO_STATE", "STRING")
    RETURN_NAMES = ("video_state", "prompt_store")
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

        # ── Annotation-only mode (no video): just return accumulated prompts ─
        if video is None and video_frames is None:
            log.info("No video/video_frames — annotation-only mode, skipping video_state creation.")
            return {
                "ui": {
                    "prompt_store": [updated_store],
                    "prompt_count": [str(len(prompts))],
                },
                "result": (None, updated_store),
            }

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
                "result": (video_state, updated_store),
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
            "result": (video_state, updated_store),
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

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "STRING")
    RETURN_NAMES = ("corrected_masks", "correction_store")
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
                "result": (masks, updated_store),
            }

        if video_frames is None:
            log.warning("video_frames not connected; corrections require the video_frames input")
            return {
                "ui": {
                    "correction_store": [updated_store],
                    "correction_count": [str(len(corrections))],
                },
                "result": (masks, updated_store),
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
            "result": (corrected, updated_store),
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
                "filename_prefix": ("STRING", {
                    "default": "sam3_propagation",
                    "tooltip": "Filename prefix. Saved as {prefix}_00001_.pt under ComfyUI output/.",
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

    def save(self, masks, video_state, filename_prefix, scores=None):
        import re
        import folder_paths

        base_dir = folder_paths.get_output_directory()

        # Support subdirectory in prefix (e.g. "video/f402ab" → output/video/f402ab_*.pt)
        prefix_dir  = os.path.dirname(filename_prefix)
        prefix_base = os.path.basename(filename_prefix)
        out_dir = os.path.join(base_dir, prefix_dir) if prefix_dir else base_dir
        os.makedirs(out_dir, exist_ok=True)

        # Find next counter (same convention as SaveVideo: prefix_00001_.pt)
        pattern = re.compile(rf"^{re.escape(prefix_base)}_(\d+)_\.pt$")
        counter = 1
        for name in os.listdir(out_dir):
            m = pattern.match(name)
            if m:
                counter = max(counter, int(m.group(1)) + 1)

        save_name = f"{prefix_base}_{counter:05d}_.pt"
        save_path = os.path.join(out_dir, save_name)

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
                    "tooltip": "Path to a .pt file, or a folder — latest .pt in the folder is loaded.",
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
        out_dir = folder_paths.get_output_directory()
        files = []
        if os.path.isdir(out_dir):
            for root, dirs, fnames in os.walk(out_dir):
                dirs.sort()
                for fname in fnames:
                    if fname.endswith(".pt"):
                        full = os.path.join(root, fname)
                        rel  = os.path.relpath(full, out_dir)
                        files.append({"path": full, "label": rel})
            files.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
        return web.json_response(files)

    @server.PromptServer.instance.routes.get("/sam3/get_paths")
    async def _get_paths_handler(request):
        import folder_paths
        return web.json_response({
            "output": folder_paths.get_output_directory(),
            "home":   os.path.expanduser("~"),
        })

    @server.PromptServer.instance.routes.get("/sam3/browse_files")
    async def _browse_files_handler(request):
        import folder_paths
        path = request.query.get("path", "")
        if not path:
            path = folder_paths.get_output_directory()
        path = os.path.normpath(os.path.realpath(path))
        if not os.path.isdir(path):
            return web.json_response({"error": "Not a directory"}, status=400)
        entries = []
        try:
            for name in sorted(os.listdir(path)):
                if name.startswith("."):
                    continue
                full = os.path.join(path, name)
                if os.path.isdir(full):
                    entries.append({"name": name, "path": full, "type": "dir"})
                elif name.endswith(".pt"):
                    try:
                        st = os.stat(full)
                        entries.append({
                            "name": name, "path": full, "type": "file",
                            "size": st.st_size, "mtime": st.st_mtime,
                        })
                    except OSError:
                        pass
        except PermissionError:
            return web.json_response({"error": "Permission denied"}, status=403)
        parent = os.path.dirname(path)
        return web.json_response({
            "current": path,
            "parent":  parent if parent != path else None,
            "entries": entries,
        })

    @server.PromptServer.instance.routes.get("/sam3/browse_dirs")
    async def _browse_dirs_handler(request):
        """Browse server-side directories (directory picker for SAM3OutputFolder)."""
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        path = request.query.get("path", "")
        if not path:
            path = output_dir
        path = os.path.normpath(os.path.realpath(path))
        if not os.path.isdir(path):
            return web.json_response({"error": "Not a directory"}, status=400)
        entries = []
        try:
            for name in sorted(os.listdir(path)):
                if name.startswith("."):
                    continue
                full = os.path.join(path, name)
                if os.path.isdir(full):
                    entries.append({"name": name, "path": full, "type": "dir"})
        except PermissionError:
            return web.json_response({"error": "Permission denied"}, status=403)
        parent = os.path.dirname(path)
        return web.json_response({
            "current":    path,
            "parent":     parent if parent != path else None,
            "entries":    entries,
            "output_dir": output_dir,
        })


# =============================================================================
# SAM3OutputFolder — set a common output subfolder for SaveVideo / SavePropagation
# =============================================================================

class SAM3OutputFolder:
    """
    Utility node: type a subfolder name once and wire it to SaveVideo and
    SAM3SavePropagation so all outputs land in the same place.

    Given folder = "f402ab":
      masks  → "f402ab/masks"  (mask video)
      visual → "f402ab/visual" (visualization video)
      propa  → "f402ab/propa"  (propagation .pt file)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "f402ab",
                    "tooltip": "Base folder name. Outputs go to output/{folder}/masks, visual, propa.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("masks", "visual", "propa", "folder_path")
    FUNCTION = "compute"
    CATEGORY = "SAM3/utils"

    def compute(self, folder):
        import folder_paths
        folder = folder.strip().strip("/")
        if not folder:
            folder = "output"
        abs_path = os.path.join(folder_paths.get_output_directory(), folder)
        return (f"{folder}/masks", f"{folder}/visual", f"{folder}/propa", abs_path)


# =============================================================================
# SAM3WritePipelineCSV — write initial tracking prompts to batch pipeline CSV
# =============================================================================

class SAM3WritePipelineCSV:
    """
    Write initial point-prompt annotations to the batch pipeline CSV.

    Step 2 annotation workflow:
      VHS_LoadVideo → SAM3AnimalPointCollector → SAM3TwoMouseTracking → SAM3WritePipelineCSV

    Reads the accumulated prompt_store from SAM3TwoMouseTracking, extracts the
    coordinates for mouse1 (obj_id=1) and mouse2 (obj_id=2), and upserts a row
    in pipeline.csv with status=prompted.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_store": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Accumulated prompts from SAM3TwoMouseTracking (prompt_store output).",
                }),
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to the source video file.",
                }),
                "video_id": ("STRING", {
                    "default": "",
                    "tooltip": "Short unique identifier for this video (e.g. f402ab).",
                }),
                "csv_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to pipeline.csv (created if it does not exist).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("csv_path",)
    FUNCTION = "write"
    CATEGORY = "SAM3/batch"
    OUTPUT_NODE = True

    def write(self, prompt_store, video_path, video_id, csv_path):
        import csv as csvmod

        HEADERS = [
            "video_id", "video_path", "status",
            "prompt_frame_idx", "mouse1_x", "mouse1_y", "mouse2_x", "mouse2_y",
            "output_pt_path", "masks_csv_path", "masks_video_path",
        ]

        try:
            prompts = json.loads(prompt_store) if str(prompt_store).strip() else []
            prompts = prompts if isinstance(prompts, list) else []
        except (json.JSONDecodeError, TypeError):
            prompts = []

        m1 = next((p for p in prompts if p.get("obj_id") == 1), None)
        m2 = next((p for p in prompts if p.get("obj_id") == 2), None)

        frame_idx = m1.get("frame_idx", "") if m1 else ""
        m1x = m1["points"][0][0] if m1 and m1.get("points") else ""
        m1y = m1["points"][0][1] if m1 and m1.get("points") else ""
        m2x = m2["points"][0][0] if m2 and m2.get("points") else ""
        m2y = m2["points"][0][1] if m2 and m2.get("points") else ""

        rows = []
        if os.path.isfile(csv_path):
            with open(csv_path, "r", newline="") as f:
                rows = list(csvmod.DictReader(f))

        new_row = {
            "video_id": video_id,
            "video_path": video_path,
            "status": "prompted",
            "prompt_frame_idx": frame_idx,
            "mouse1_x": m1x,
            "mouse1_y": m1y,
            "mouse2_x": m2x,
            "mouse2_y": m2y,
            "output_pt_path": "",
            "masks_csv_path": "",
            "masks_video_path": "",
        }

        replaced = False
        for i, row in enumerate(rows):
            if row.get("video_id") == video_id:
                new_row["output_pt_path"]   = row.get("output_pt_path", "")
                new_row["masks_csv_path"]   = row.get("masks_csv_path", "")
                new_row["masks_video_path"] = row.get("masks_video_path", "")
                rows[i] = new_row
                replaced = True
                break
        if not replaced:
            rows.append(new_row)

        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csvmod.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(rows)

        log.info(f"[SAM3WritePipelineCSV] {csv_path}: video_id={video_id} status=prompted "
                 f"frame={frame_idx} m1=({m1x},{m1y}) m2=({m2x},{m2y})")

        return {
            "ui": {"csv_path": [csv_path]},
            "result": (csv_path,),
        }


# =============================================================================
# SAM3WriteCorrectionsCSV — write correction prompts to batch corrections CSV
# =============================================================================

class SAM3WriteCorrectionsCSV:
    """
    Write correction-prompt annotations to the batch corrections CSV.

    Step 4 correction workflow:
      SAM3LoadPropagation → SAM3FrameCorrector → SAM3WriteCorrectionsCSV
                                  ↑
                        SAM3AnimalPointCollector

    Reads the accumulated correction_store from SAM3FrameCorrector, expands each
    entry to per-point rows, and writes them to corrections.csv.  Also updates
    pipeline.csv status to "corrected" for this video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "correction_store": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Accumulated corrections from SAM3FrameCorrector (correction_store output).",
                }),
                "video_id": ("STRING", {
                    "default": "",
                    "tooltip": "Short unique identifier matching a row in pipeline.csv.",
                }),
                "corrections_csv_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to corrections.csv (created if it does not exist).",
                }),
                "pipeline_csv_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to pipeline.csv (status updated to corrected).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("corrections_csv_path",)
    FUNCTION = "write"
    CATEGORY = "SAM3/batch"
    OUTPUT_NODE = True

    def write(self, correction_store, video_id, corrections_csv_path, pipeline_csv_path):
        import csv as csvmod

        CORR_HEADERS = ["video_id", "frame_idx", "mouse_id", "point_type", "x", "y"]
        PIPE_HEADERS = [
            "video_id", "video_path", "gt_csv_path", "status",
            "prompt_frame_idx", "mouse1_x", "mouse1_y", "mouse2_x", "mouse2_y",
            "output_pt_path", "masks_csv_path", "masks_video_path",
        ]

        try:
            corrections = json.loads(correction_store) if str(correction_store).strip() else []
            corrections = corrections if isinstance(corrections, list) else []
        except (json.JSONDecodeError, TypeError):
            corrections = []

        # Load existing corrections (drop old rows for this video)
        existing = []
        if os.path.isfile(corrections_csv_path):
            with open(corrections_csv_path, "r", newline="") as f:
                existing = [r for r in csvmod.DictReader(f) if r.get("video_id") != video_id]

        new_rows = []
        for entry in corrections:
            frame_idx = entry.get("frame_idx", "")
            obj_id    = entry.get("obj_id", 1)
            points    = entry.get("points", [])
            labels    = entry.get("labels", [])
            for pt, label in zip(points, labels):
                new_rows.append({
                    "video_id":   video_id,
                    "frame_idx":  frame_idx,
                    "mouse_id":   obj_id,
                    "point_type": "pos" if label == 1 else "neg",
                    "x":          pt[0],
                    "y":          pt[1],
                })

        corr_dir = os.path.dirname(os.path.abspath(corrections_csv_path))
        os.makedirs(corr_dir, exist_ok=True)
        with open(corrections_csv_path, "w", newline="") as f:
            writer = csvmod.DictWriter(f, fieldnames=CORR_HEADERS)
            writer.writeheader()
            writer.writerows(existing + new_rows)

        # Update pipeline CSV status → corrected
        if pipeline_csv_path and os.path.isfile(pipeline_csv_path) and new_rows:
            pipe_rows = []
            with open(pipeline_csv_path, "r", newline="") as f:
                pipe_rows = list(csvmod.DictReader(f))
            for row in pipe_rows:
                if row.get("video_id") == video_id:
                    row["status"] = "corrected"
            if pipe_rows:
                headers = list(pipe_rows[0].keys())
                with open(pipeline_csv_path, "w", newline="") as f:
                    writer = csvmod.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(pipe_rows)

        log.info(f"[SAM3WriteCorrectionsCSV] {corrections_csv_path}: "
                 f"video_id={video_id} {len(new_rows)} correction points written")

        return {
            "ui": {"corrections_csv_path": [corrections_csv_path]},
            "result": (corrections_csv_path,),
        }


# =============================================================================
# SAM3WorkingDirSelector — batch pipeline working directory + video selector
# =============================================================================

class SAM3WorkingDirSelector:
    """
    Pipeline CSV-based video selector for the batch pipeline.

    Reads pipeline.csv from working_dir and presents video_id values for
    selection. The annotator sees only video_ids (not raw file paths).
    video_path is resolved by looking up the selected video_id in pipeline.csv.

    Outputs feed into:
      video_path  → VHS_LoadVideoPath
      video_id    → SAM3WritePipelineCSV / SAM3WriteCorrectionsCSV
      csv_path    → SAM3WritePipelineCSV (pipeline.csv)
      corrections_csv_path → SAM3WriteCorrectionsCSV
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "working_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Root working directory containing pipeline.csv",
                }),
                "selected_video_id": ("STRING", {
                    "default": "",
                    "tooltip": "video_id selected from pipeline.csv — set by the widget dropdown.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "video_id", "csv_path", "corrections_csv_path", "working_dir", "output_pt_path", "masks_video_path")
    FUNCTION = "select"
    CATEGORY = "SAM3/batch"

    def select(self, working_dir, selected_video_id):
        import pathlib, csv as _csv
        wd = pathlib.Path(working_dir.strip())
        csv_path = str(wd / "pipeline.csv")
        corrections_csv_path = str(wd / "corrections.csv")
        video_path = ""
        output_pt_path = ""
        masks_video_path = ""
        if selected_video_id and (wd / "pipeline.csv").exists():
            with open(csv_path, newline="") as f:
                for row in _csv.DictReader(f):
                    if row.get("video_id") == selected_video_id:
                        video_path = row.get("video_path", "")
                        output_pt_path = row.get("output_pt_path", "")
                        masks_video_path = row.get("masks_video_path", "")
                        break
        return (video_path, selected_video_id, csv_path, corrections_csv_path, str(wd), output_pt_path, masks_video_path)


if _SERVER_AVAILABLE:
    @server.PromptServer.instance.routes.get("/sam3/wd/list_pipeline_videos")
    async def _wd_list_pipeline_videos(request):
        import csv as _csv
        working_dir = request.query.get("working_dir", "").strip()
        csv_path = os.path.join(working_dir, "pipeline.csv")
        videos = []
        if os.path.isfile(csv_path):
            with open(csv_path, newline="") as f:
                for row in _csv.DictReader(f):
                    videos.append({
                        "video_id": row.get("video_id", ""),
                        "status":   row.get("status", ""),
                    })
        return web.json_response({"videos": videos})


# =============================================================================
# Node Mappings
# =============================================================================

# ---------------------------------------------------------------------------
# Patch VHS_LoadVideo.VALIDATE_INPUTS to handle None (linked input).
# When video is a linked output from another node, ComfyUI passes None during
# validation. The stock implementation calls None.endswith() and crashes.
# ---------------------------------------------------------------------------
def _patch_vhs_load_video():
    try:
        import folder_paths
        from nodes import NODE_CLASS_MAPPINGS as _NCM
        cls = _NCM.get("VHS_LoadVideo")
        if cls is None:
            return
        orig = cls.__dict__.get("VALIDATE_INPUTS")
        if orig is None:
            return

        @classmethod  # type: ignore[misc]
        def _safe_validate(klass, video, **kwargs):
            if video is None:
                return True  # linked input — skip validation
            return orig.__func__(klass, video, **kwargs)

        cls.VALIDATE_INPUTS = _safe_validate
        log.info("[sam3] Patched VHS_LoadVideo.VALIDATE_INPUTS to handle linked (None) input")
    except Exception as e:
        log.warning(f"[sam3] Could not patch VHS_LoadVideo: {e}")

_patch_vhs_load_video()

# =============================================================================

# =============================================================================
# SAM3FrameLoader — load a single frame from a video file using cv2
# =============================================================================

class SAM3FrameLoader:
    """
    Load a single frame directly from a video file using cv2.
    Much faster than VHS_LoadVideo for annotation workflows where only
    one specific frame is needed.

    Usage in batch_annotation workflow:
      SAM3FrameLoader → SAM3AnimalPointCollector
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to the video file.",
                }),
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "Frame index to load (0-based).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_frame"
    CATEGORY = "SAM3/batch"

    def load_frame(self, video_path: str, frame_idx: int):
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[SAM3FrameLoader] Cannot open video: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret:
            raise RuntimeError(
                f"[SAM3FrameLoader] Failed to read frame {frame_idx} from {video_path}"
            )

        # cv2 returns BGR uint8 — convert to RGB float32 [0,1], shape (1, H, W, C)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        return (tensor.unsqueeze(0),)


# =============================================================================
# SAM3BuildPromptStore — combine two single prompts into a prompt_store JSON
# =============================================================================

class SAM3BuildPromptStore:
    """
    Combine mouse1 and mouse2 single prompts into a prompt_store JSON string
    for SAM3WritePipelineCSV.  No accumulation — builds a fresh store every run.

    Usage in batch_annotation workflow:
      Collector1 (animal_id=1) ─┐
                                 ├→ SAM3BuildPromptStore → prompt_store → SAM3WritePipelineCSV
      Collector2 (animal_id=2) ─┘
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mouse1_prompt": ("SAM3_SINGLE_PROMPT", {
                    "tooltip": "Single prompt from SAM3AnimalPointCollector (animal_id=1).",
                }),
                "mouse2_prompt": ("SAM3_SINGLE_PROMPT", {
                    "tooltip": "Single prompt from SAM3AnimalPointCollector (animal_id=2).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_store",)
    FUNCTION = "build"
    CATEGORY = "SAM3/batch"

    def build(self, mouse1_prompt, mouse2_prompt):
        prompts = []
        if mouse1_prompt and mouse1_prompt.get("points"):
            prompts.append(mouse1_prompt)
        if mouse2_prompt and mouse2_prompt.get("points"):
            prompts.append(mouse2_prompt)
        return (json.dumps(prompts),)


NODE_CLASS_MAPPINGS = {
    "SAM3TwoMouseTracking": SAM3TwoMouseTracking,
    "SAM3FrameCorrector": SAM3FrameCorrector,
    "SAM3SavePropagation": SAM3SavePropagation,
    "SAM3LoadPropagation": SAM3LoadPropagation,
    "SAM3OutputFolder": SAM3OutputFolder,
    "SAM3WritePipelineCSV": SAM3WritePipelineCSV,
    "SAM3WriteCorrectionsCSV": SAM3WriteCorrectionsCSV,
    "SAM3WorkingDirSelector": SAM3WorkingDirSelector,
    "SAM3FrameLoader": SAM3FrameLoader,
    "SAM3BuildPromptStore": SAM3BuildPromptStore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3TwoMouseTracking": "SAM3 Two-Mouse Tracking",
    "SAM3FrameCorrector": "SAM3 Frame Corrector",
    "SAM3SavePropagation": "SAM3 Save Propagation",
    "SAM3LoadPropagation": "SAM3 Load Propagation",
    "SAM3OutputFolder": "SAM3 Output Folder",
    "SAM3WritePipelineCSV": "SAM3 Write Pipeline CSV",
    "SAM3WriteCorrectionsCSV": "SAM3 Write Corrections CSV",
    "SAM3WorkingDirSelector": "SAM3 Working Dir Selector",
    "SAM3FrameLoader": "SAM3 Frame Loader",
    "SAM3BuildPromptStore": "SAM3 Build Prompt Store",
}
