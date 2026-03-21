"""
SAM3 Video Prompt Nodes

Composable prompt-entry nodes for video tracking.

Workflow:
  SAM3VideoPromptPoint(frame=0, obj=1) ──────────────────────────────────┐
  SAM3VideoPromptPoint(frame=0, obj=2) ────► SAM3VideoPromptPoint(chain) ─► SAM3BuildVideoState ─► SAM3Propagate
  SAM3VideoPromptPoint(frame=70, obj=2) ───────────────────────────────────►/

Each SAM3VideoPromptPoint node stores one (frame_idx, obj_id, points) entry.
Chain them by connecting prompt_list outputs into the next node's prompt_list input.
SAM3BuildVideoState collects all prompts and initialises the video state.
"""
import hashlib
import json
import io
import base64
import logging
import os

import numpy as np
from PIL import Image

log = logging.getLogger("sam3")

from .video_state import (
    VideoPrompt,
    VideoConfig,
    create_video_state,
    create_video_state_from_file,
)
from .utils import print_mem


# ============================================================================
# SAM3VideoPromptPoint
# ============================================================================

class SAM3VideoPromptPoint:
    """
    One video point-prompt entry: (frame_idx, obj_id, points).

    • Left-click  → positive point (green)
    • Shift/Right-click → negative point (red)

    Connect prompt_list output to the next SAM3VideoPromptPoint (or to
    SAM3BuildVideoState) to accumulate prompts from multiple frames/objects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Frame to click on. Use Select Images to pick the correct frame_idx."
                }),
                "points_store":    ("STRING", {"multiline": False, "default": "{}"}),
                "coordinates":     ("STRING", {"multiline": False, "default": "[]"}),
                "neg_coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index where these points are applied. Must match the frame shown in 'image'."
                }),
                "obj_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "tooltip": "Object ID (1=mouse1, 2=mouse2, …). Each tracked object needs a unique ID."
                }),
            },
            "optional": {
                "prompt_list": ("SAM3_PROMPT_LIST", {
                    "tooltip": "Chain from a previous SAM3VideoPromptPoint to accumulate prompts."
                }),
            },
        }

    RETURN_TYPES = ("SAM3_PROMPT_LIST",)
    RETURN_NAMES = ("prompt_list",)
    FUNCTION = "add_prompt"
    CATEGORY = "SAM3/video"
    OUTPUT_NODE = True  # keeps canvas alive even without downstream connections

    @classmethod
    def IS_CHANGED(cls, image, points_store, coordinates, neg_coordinates,
                   frame_idx, obj_id, prompt_list=None):
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        h.update(str(frame_idx).encode())
        h.update(str(obj_id).encode())
        return h.hexdigest()

    def add_prompt(self, image, points_store, coordinates, neg_coordinates,
                   frame_idx, obj_id, prompt_list=None):
        # Parse point lists from hidden widgets
        try:
            pos_coords = json.loads(coordinates) if coordinates.strip() else []
            neg_coords = json.loads(neg_coordinates) if neg_coordinates.strip() else []
        except json.JSONDecodeError:
            pos_coords, neg_coords = [], []

        img_height, img_width = image.shape[1], image.shape[2]

        pts, labels = [], []
        for p in pos_coords:
            pts.append([float(p["x"]), float(p["y"])])
            labels.append(1)
        for p in neg_coords:
            pts.append([float(p["x"]), float(p["y"])])
            labels.append(0)

        # Accumulate
        prompts = list(prompt_list) if prompt_list else []

        if pts:
            prompts.append({
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "points": pts,       # pixel coords in 'image' space
                "labels": labels,
                "img_width": img_width,
                "img_height": img_height,
            })
            log.info(f"SAM3VideoPromptPoint: frame={frame_idx} obj={obj_id} "
                     f"{sum(l==1 for l in labels)} pos, {sum(l==0 for l in labels)} neg")
        else:
            log.warning(f"SAM3VideoPromptPoint: frame={frame_idx} obj={obj_id} — no points clicked")

        img_base64 = self._tensor_to_base64(image)
        return {
            "ui":     {"bg_image": [img_base64]},
            "result": (prompts,),
        }

    @staticmethod
    def _tensor_to_base64(tensor):
        arr = tensor[0].cpu().numpy()
        arr = (arr * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================================
# SAM3BuildVideoState
# ============================================================================

class SAM3BuildVideoState:
    """
    Build SAM3_VIDEO_STATE from a video and an accumulated SAM3_PROMPT_LIST.

    Connect the final SAM3VideoPromptPoint's prompt_list output here,
    then feed the resulting video_state into SAM3Propagate.
    """

    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch [N, H, W, C]"
                }),
                "video": ("VIDEO", {
                    "tooltip": "Video from Load Video node. Takes priority over video_frames."
                }),
                "prompt_list": ("SAM3_PROMPT_LIST", {
                    "tooltip": "Accumulated prompts from SAM3VideoPromptPoint chain."
                }),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "build"
    CATEGORY = "SAM3/video"

    @classmethod
    def IS_CHANGED(cls, score_threshold, video_frames=None, video=None, prompt_list=None):
        h = hashlib.md5()
        h.update(str(score_threshold).encode())
        if video is not None:
            try:
                source = video.get_stream_source()
                if isinstance(source, str):
                    h.update(source.encode())
                    try:
                        h.update(str(os.path.getmtime(source)).encode())
                    except OSError:
                        pass
                else:
                    h.update(str(video.get_frame_count()).encode())
            except Exception:
                h.update(str(id(video)).encode())
        elif video_frames is not None:
            h.update(str(video_frames.shape).encode())
        h.update(str(prompt_list).encode() if prompt_list else b"none")
        return h.hexdigest()

    def build(self, score_threshold, video_frames=None, video=None, prompt_list=None):
        if video is None and video_frames is None:
            raise ValueError("Either video or video_frames must be provided.")

        # Cache key
        h = hashlib.md5()
        h.update(str(score_threshold).encode())
        if video is not None:
            try:
                source = video.get_stream_source()
                h.update(str(source).encode())
            except Exception:
                h.update(str(id(video)).encode())
        elif video_frames is not None:
            h.update(str(video_frames.shape).encode())
        h.update(str(id(prompt_list)).encode() if prompt_list else b"none")
        cache_key = h.hexdigest()

        if cache_key in SAM3BuildVideoState._cache:
            log.info(f"CACHE HIT - SAM3BuildVideoState key={cache_key[:8]}")
            return (SAM3BuildVideoState._cache[cache_key],)

        log.info(f"CACHE MISS - building video state key={cache_key[:8]}")
        print_mem("Before SAM3BuildVideoState")

        # Create base video state
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

        # Add prompts
        if prompt_list:
            for entry in prompt_list:
                frame_idx  = entry["frame_idx"]
                obj_id     = entry["obj_id"]
                pts_px     = entry["points"]   # pixel coords in prompt image space
                labels     = entry["labels"]
                img_w      = entry["img_width"]
                img_h      = entry["img_height"]

                # Normalise to [0, 1] using the frame dims from which points were taken
                pts_norm = [[x / img_w, y / img_h] for x, y in pts_px]

                prompt = VideoPrompt.create_point(frame_idx, obj_id, pts_norm, labels)
                video_state = video_state.with_prompt(prompt)

                pos = sum(l == 1 for l in labels)
                neg = sum(l == 0 for l in labels)
                log.info(f"  Added: frame={frame_idx} obj={obj_id} "
                         f"{pos} pos, {neg} neg points")
        else:
            log.warning("SAM3BuildVideoState: prompt_list is empty — no prompts added")

        print_mem("After SAM3BuildVideoState")
        SAM3BuildVideoState._cache[cache_key] = video_state
        return (video_state,)


# ============================================================================
# Node Mappings
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM3VideoPromptPoint": SAM3VideoPromptPoint,
    "SAM3BuildVideoState":  SAM3BuildVideoState,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoPromptPoint": "SAM3 Video Prompt Point",
    "SAM3BuildVideoState":  "SAM3 Build Video State",
}
