"""
SAM3 Interactive Collectors — Point, BBox, Multi-Region, and Interactive Segmentation

Point/BBox editor widgets adapted from ComfyUI-KJNodes
Original: https://github.com/kijai/ComfyUI-KJNodes
Author: kijai
License: Apache 2.0
"""

import asyncio
import gc
import hashlib
import logging
import json
import io
import base64
import threading

import numpy as np
import torch
from PIL import Image

try:
    import server
    from aiohttp import web
    _SERVER_AVAILABLE = True
except Exception:
    server = None
    _SERVER_AVAILABLE = False
from .utils import comfy_image_to_pil, visualize_masks_on_image, masks_to_comfy_mask, pil_to_comfy_image

log = logging.getLogger("sam3")

# ---------------------------------------------------------------------------
# Interactive segmentation cache — keyed by node unique_id
# ---------------------------------------------------------------------------
_INTERACTIVE_CACHE = {}

# ---------------------------------------------------------------------------
# SAM3AnimalPointCollector video frame cache — keyed by node unique_id
# ---------------------------------------------------------------------------
_POINT_COLLECTOR_FRAMES = {}         # node_id -> video_frames tensor [N, H, W, C]
_POINT_COLLECTOR_MASK_PATHS = {}     # node_id -> mask video file path (str)
_POINT_COLLECTOR_MASK_TENSORS = {}   # node_id -> mask_frames tensor [N, H, W, C]

# Serializes GPU work from parallel per-prompt requests
_SEGMENT_LOCK = threading.Lock()


class SAM3AnimalPointCollector:
    """
    Interactive Point Collector for SAM3

    Displays image canvas in the node where users can click to add:
    - Positive points (Left-click) - green circles
    - Negative points (Shift+Left-click or Right-click) - red circles

    Upgraded: connect video_frames + set frame_idx/animal_id to navigate frames
    without a separate Select Images node. Outputs SAM3_SINGLE_PROMPT for use
    with SAM3TwoMouseTracking accumulator.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points_store": ("STRING", {"multiline": False, "default": "{}"}),
                "coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "frame_idx": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Frame to display and annotate. Synced with the video slider."
                }),
                "animal_id": ("INT", {
                    "default": 1, "min": 1, "max": 32,
                    "tooltip": "Animal ID for this prompt (1, 2, …). Used as obj_id in SAM3."
                }),
            },
            "optional": {
                "video_frames": ("IMAGE", {
                    "tooltip": "All video frames as batch [N,H,W,C]. Enables video playback slider."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Single image fallback when video_frames is not connected."
                }),
                "mask_frames": ("IMAGE", {
                    "tooltip": "Visualization frames from SAM3VideoOutput. Shown as overlay when 'Mask' toggle is on. Takes priority over mask_video_path.",
                }),
                "mask_video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Fallback: path to mask video file. Used only when mask_frames is not connected.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT", "SAM3_SINGLE_PROMPT")
    RETURN_NAMES = ("positive_points", "negative_points", "single_prompt")
    FUNCTION = "collect_points"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(cls, points_store, coordinates, neg_coordinates,
                   frame_idx=0, animal_id=1, video_frames=None, image=None,
                   mask_frames=None, mask_video_path=None, unique_id=None):
        h = hashlib.md5()
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        h.update(str(frame_idx).encode())
        h.update(str(animal_id).encode())
        if video_frames is not None:
            h.update(str(video_frames.shape).encode())
        elif image is not None:
            h.update(str(image.shape).encode())
        if mask_frames is not None:
            h.update(str(mask_frames.shape).encode())
        elif mask_video_path:
            h.update(mask_video_path.encode())
        return h.hexdigest()

    def collect_points(self, points_store, coordinates, neg_coordinates,
                       frame_idx=0, animal_id=1, video_frames=None, image=None,
                       mask_frames=None, mask_video_path=None, unique_id=None):
        # Select the display frame
        if video_frames is not None:
            idx = min(frame_idx, video_frames.shape[0] - 1)
            display = video_frames[idx:idx+1]   # [1, H, W, C]
        elif image is not None:
            display = image
        else:
            raise ValueError("Either video_frames or image must be provided.")

        # Cache key
        h = hashlib.md5()
        h.update(str(display.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        h.update(str(frame_idx).encode())
        h.update(str(animal_id).encode())
        if mask_frames is not None:
            h.update(str(mask_frames.shape).encode())
        elif mask_video_path:
            h.update(mask_video_path.encode())
        cache_key = h.hexdigest()

        img_base64 = self.tensor_to_base64(display)

        # Helper to build full UI dict (used by both cache-hit and cache-miss paths)
        def _build_ui(total_frames_n):
            has_mask = mask_frames is not None or bool(mask_video_path and mask_video_path.strip())
            return {
                "bg_image":       [img_base64],
                "node_id":        [str(unique_id) if unique_id else ""],
                "total_frames":   [str(total_frames_n)],
                "fps":            ["4"],
                "frame_idx":      [str(frame_idx)],
                "has_mask_video": ["1" if has_mask else "0"],
            }

        if cache_key in SAM3AnimalPointCollector._cache:
            log.info(f"CACHE HIT - SAM3PointCollector key={cache_key[:8]}")
            # Still update mask path cache and total_frames for UI
            tf = 1
            if unique_id is not None:
                if video_frames is not None:
                    _POINT_COLLECTOR_FRAMES[str(unique_id)] = video_frames
                    tf = video_frames.shape[0]
                elif image is not None:
                    _POINT_COLLECTOR_FRAMES[str(unique_id)] = image
                    tf = image.shape[0]
                if mask_frames is not None:
                    _POINT_COLLECTOR_MASK_TENSORS[str(unique_id)] = mask_frames
                    _POINT_COLLECTOR_MASK_PATHS.pop(str(unique_id), None)
                elif mask_video_path and mask_video_path.strip():
                    _POINT_COLLECTOR_MASK_PATHS[str(unique_id)] = mask_video_path.strip()
                    _POINT_COLLECTOR_MASK_TENSORS.pop(str(unique_id), None)
                else:
                    _POINT_COLLECTOR_MASK_PATHS.pop(str(unique_id), None)
                    _POINT_COLLECTOR_MASK_TENSORS.pop(str(unique_id), None)
            return {
                "ui": _build_ui(tf),
                "result": SAM3AnimalPointCollector._cache[cache_key],
            }

        log.info(f"CACHE MISS - SAM3PointCollector key={cache_key[:8]}")

        try:
            pos_coords = json.loads(coordinates) if coordinates and coordinates.strip() else []
            neg_coords = json.loads(neg_coordinates) if neg_coordinates and neg_coordinates.strip() else []
        except json.JSONDecodeError:
            pos_coords, neg_coords = [], []

        img_height, img_width = display.shape[1], display.shape[2]
        log.info(f"frame={frame_idx} animal={animal_id} img={img_width}x{img_height} "
                 f"pos={len(pos_coords)} neg={len(neg_coords)}")

        positive_points = {"points": [], "labels": []}
        negative_points = {"points": [], "labels": []}

        for p in pos_coords:
            nx, ny = p['x'] / img_width, p['y'] / img_height
            positive_points["points"].append([nx, ny])
            positive_points["labels"].append(1)

        for n in neg_coords:
            nx, ny = n['x'] / img_width, n['y'] / img_height
            negative_points["points"].append([nx, ny])
            negative_points["labels"].append(0)

        # Build single_prompt (pixel coords + metadata for accumulator)
        pts_px, labels = [], []
        for p in pos_coords:
            pts_px.append([float(p['x']), float(p['y'])])
            labels.append(1)
        for n in neg_coords:
            pts_px.append([float(n['x']), float(n['y'])])
            labels.append(0)

        single_prompt = None
        if pts_px:
            single_prompt = {
                "frame_idx": frame_idx,
                "obj_id":    animal_id,
                "points":    pts_px,
                "labels":    labels,
                "img_width":  img_width,
                "img_height": img_height,
            }

        result = (positive_points, negative_points, single_prompt)
        SAM3AnimalPointCollector._cache[cache_key] = result

        # Cache video frames for on-demand frame serving
        total_frames = 1
        if unique_id is not None:
            if video_frames is not None:
                _POINT_COLLECTOR_FRAMES[str(unique_id)] = video_frames
                total_frames = video_frames.shape[0]
            elif image is not None:
                _POINT_COLLECTOR_FRAMES[str(unique_id)] = image
                total_frames = image.shape[0]
            # Cache mask source (tensor takes priority over file path)
            if mask_frames is not None:
                _POINT_COLLECTOR_MASK_TENSORS[str(unique_id)] = mask_frames
                _POINT_COLLECTOR_MASK_PATHS.pop(str(unique_id), None)
            elif mask_video_path and mask_video_path.strip():
                _POINT_COLLECTOR_MASK_PATHS[str(unique_id)] = mask_video_path.strip()
                _POINT_COLLECTOR_MASK_TENSORS.pop(str(unique_id), None)
            else:
                _POINT_COLLECTOR_MASK_PATHS.pop(str(unique_id), None)
                _POINT_COLLECTOR_MASK_TENSORS.pop(str(unique_id), None)

        return {
            "ui": _build_ui(total_frames),
            "result": result,
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        img_array = tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')



class SAM3PointCollector:
    """
    Interactive Point Collector for SAM3

    Displays image canvas in the node where users can click to add:
    - Positive points (Left-click) - green circles
    - Negative points (Shift+Left-click or Right-click) - red circles

    Outputs point arrays to feed into SAM3Segmentation node.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to display in interactive canvas. Left-click to add positive points (green), Shift+Left-click or Right-click to add negative points (red). Points are automatically normalized to image dimensions."
                }),
                "points_store": ("STRING", {"multiline": False, "default": "{}"}),
                "coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_coordinates": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT")
    RETURN_NAMES = ("positive_points", "negative_points")
    FUNCTION = "collect_points"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(cls, image, points_store, coordinates, neg_coordinates):
        # Return hash based on actual point content, not object identity
        # This ensures downstream nodes don't re-run when points haven't changed
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        result = h.hexdigest()
        log.debug(f"IS_CHANGED SAM3PointCollector: shape={image.shape}, coords={coordinates}, neg_coords={neg_coordinates}")
        log.debug(f"IS_CHANGED SAM3PointCollector: returning hash={result}")
        return result

    def collect_points(self, image, points_store, coordinates, neg_coordinates):
        """
        Collect points from interactive canvas

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            points_store: Combined JSON storage (hidden widget)
            coordinates: Positive points JSON (hidden widget)
            neg_coordinates: Negative points JSON (hidden widget)

        Returns:
            Tuple of (positive_points, negative_points) as separate SAM3_POINTS_PROMPT outputs
        """
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3PointCollector._cache:
            cached = SAM3PointCollector._cache[cache_key]
            log.info(f"CACHE HIT - returning cached result for key={cache_key[:8]}")
            # Still need to return UI update
            img_base64 = self.tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_base64]},
                "result": cached  # Return the SAME objects
            }

        log.info(f"CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse coordinates from JSON
        try:
            pos_coords = json.loads(coordinates) if coordinates and coordinates.strip() else []
            neg_coords = json.loads(neg_coordinates) if neg_coordinates and neg_coordinates.strip() else []
        except json.JSONDecodeError:
            pos_coords = []
            neg_coords = []

        log.info(f"Collected {len(pos_coords)} positive, {len(neg_coords)} negative points")

        # Get image dimensions for normalization
        img_height, img_width = image.shape[1], image.shape[2]
        log.info(f"Image dimensions: {img_width}x{img_height}")

        # Convert to SAM3 point format - separate positive and negative outputs
        # SAM3 expects normalized coordinates (0-1), so divide by image dimensions
        positive_points = {"points": [], "labels": []}
        negative_points = {"points": [], "labels": []}

        # Add positive points (label = 1) - normalize to 0-1
        for p in pos_coords:
            normalized_x = p['x'] / img_width
            normalized_y = p['y'] / img_height
            positive_points["points"].append([normalized_x, normalized_y])
            positive_points["labels"].append(1)
            log.info(f"  Positive point: ({p['x']:.1f}, {p['y']:.1f}) -> ({normalized_x:.3f}, {normalized_y:.3f})")

        # Add negative points (label = 0) - normalize to 0-1
        for n in neg_coords:
            normalized_x = n['x'] / img_width
            normalized_y = n['y'] / img_height
            negative_points["points"].append([normalized_x, normalized_y])
            negative_points["labels"].append(0)
            log.info(f"  Negative point: ({n['x']:.1f}, {n['y']:.1f}) -> ({normalized_x:.3f}, {normalized_y:.3f})")

        log.info(f"Output: {len(positive_points['points'])} positive, {len(negative_points['points'])} negative")

        # Cache the result
        result = (positive_points, negative_points)
        SAM3PointCollector._cache[cache_key] = result

        # Send image back to widget as base64
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": result
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        img_array = tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


class SAM3BBoxCollector:
    """
    Interactive BBox Collector for SAM3

    Displays image canvas in the node where users can click and drag to add:
    - Positive bounding boxes (Left-click and drag) - cyan rectangles
    - Negative bounding boxes (Shift+Left-click and drag or Right-click and drag) - red rectangles

    Outputs bbox arrays to feed into SAM3Segmentation node.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to display in interactive canvas. Click and drag to draw positive bboxes (cyan), Shift+Click/Right-click and drag to draw negative bboxes (red). Bounding boxes are automatically normalized to image dimensions."
                }),
                "bboxes": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_bboxes": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("positive_bboxes", "negative_bboxes")
    FUNCTION = "collect_bboxes"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(cls, image, bboxes, neg_bboxes):
        # Return hash based on actual bbox content, not object identity
        # This ensures downstream nodes don't re-run when bboxes haven't changed
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(bboxes.encode())
        h.update(neg_bboxes.encode())
        result = h.hexdigest()
        log.debug(f"IS_CHANGED SAM3BBoxCollector: shape={image.shape}, bboxes={bboxes}, neg_bboxes={neg_bboxes}")
        log.debug(f"IS_CHANGED SAM3BBoxCollector: returning hash={result}")
        return result

    def collect_bboxes(self, image, bboxes, neg_bboxes):
        """
        Collect bounding boxes from interactive canvas

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            bboxes: Positive BBoxes JSON array (hidden widget)
            neg_bboxes: Negative BBoxes JSON array (hidden widget)

        Returns:
            Tuple of (positive_bboxes, negative_bboxes) as separate SAM3_BOXES_PROMPT outputs
        """
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(bboxes.encode())
        h.update(neg_bboxes.encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3BBoxCollector._cache:
            cached = SAM3BBoxCollector._cache[cache_key]
            log.info(f"CACHE HIT - returning cached result for key={cache_key[:8]}")
            # Still need to return UI update
            img_base64 = self.tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_base64]},
                "result": cached  # Return the SAME objects
            }

        log.info(f"CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse bboxes from JSON
        try:
            pos_bbox_list = json.loads(bboxes) if bboxes and bboxes.strip() else []
            neg_bbox_list = json.loads(neg_bboxes) if neg_bboxes and neg_bboxes.strip() else []
        except json.JSONDecodeError:
            pos_bbox_list = []
            neg_bbox_list = []

        log.info(f"Collected {len(pos_bbox_list)} positive, {len(neg_bbox_list)} negative bboxes")

        # Get image dimensions for normalization
        img_height, img_width = image.shape[1], image.shape[2]
        log.info(f"Image dimensions: {img_width}x{img_height}")

        # Convert to SAM3_BOXES_PROMPT format with boxes and labels
        positive_boxes = []
        positive_labels = []
        negative_boxes = []
        negative_labels = []

        # Add positive bboxes (label = True)
        for bbox in pos_bbox_list:
            # Normalize bbox coordinates to 0-1 range
            x1_norm = bbox['x1'] / img_width
            y1_norm = bbox['y1'] / img_height
            x2_norm = bbox['x2'] / img_width
            y2_norm = bbox['y2'] / img_height

            # Convert from [x1, y1, x2, y2] to [center_x, center_y, width, height]
            # SAM3 expects boxes in center format
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            width = x2_norm - x1_norm
            height = y2_norm - y1_norm

            positive_boxes.append([center_x, center_y, width, height])
            positive_labels.append(True)  # Positive boxes
            log.info(f"  Positive BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}) -> center=({center_x:.3f}, {center_y:.3f}) size=({width:.3f}, {height:.3f})")

        # Add negative bboxes (label = False)
        for bbox in neg_bbox_list:
            # Normalize bbox coordinates to 0-1 range
            x1_norm = bbox['x1'] / img_width
            y1_norm = bbox['y1'] / img_height
            x2_norm = bbox['x2'] / img_width
            y2_norm = bbox['y2'] / img_height

            # Convert from [x1, y1, x2, y2] to [center_x, center_y, width, height]
            # SAM3 expects boxes in center format
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            width = x2_norm - x1_norm
            height = y2_norm - y1_norm

            negative_boxes.append([center_x, center_y, width, height])
            negative_labels.append(False)  # Negative boxes
            log.info(f"  Negative BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}) -> center=({center_x:.3f}, {center_y:.3f}) size=({width:.3f}, {height:.3f})")

        log.info(f"Output: {len(positive_boxes)} positive, {len(negative_boxes)} negative bboxes")

        # Format as SAM3_BOXES_PROMPT (dict with 'boxes' and 'labels' keys)
        positive_prompt = {
            "boxes": positive_boxes,
            "labels": positive_labels
        }
        negative_prompt = {
            "boxes": negative_boxes,
            "labels": negative_labels
        }

        # Cache the result
        result = (positive_prompt, negative_prompt)
        SAM3BBoxCollector._cache[cache_key] = result

        # Send image back to widget as base64
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": result
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        # Convert from [B, H, W, C] to PIL Image
        # Take first image if batch
        img_array = tensor[0].cpu().numpy()
        # Convert from 0-1 float to 0-255 uint8
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64


class SAM3MultiRegionCollector:
    """
    Interactive Multi-Region Collector for SAM3

    Displays image canvas in the node where users can:
    - Click/Right-click: Add positive/negative POINTS
    - Shift + Click/Drag: Add positive/negative BOXES

    Supports multiple prompt regions via tab bar.
    Each prompt region has its own set of points and boxes.

    Outputs a list of prompts for multi-object segmentation.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to display in interactive canvas. Click to add points, Shift+drag to draw boxes. Use tab bar to manage multiple prompt regions."
                }),
                "multi_prompts_store": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_MULTI_PROMPTS",)
    RETURN_NAMES = ("multi_prompts",)
    FUNCTION = "collect_prompts"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, image, multi_prompts_store):
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(multi_prompts_store.encode())
        return h.hexdigest()

    def collect_prompts(self, image, multi_prompts_store):
        """
        Collect multiple prompt regions from interactive canvas.

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            multi_prompts_store: JSON string containing all prompt regions

        Returns:
            List of prompt dicts, each with positive/negative points/boxes
        """
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(multi_prompts_store.encode())
        cache_key = h.hexdigest()

        # Check cache
        if cache_key in SAM3MultiRegionCollector._cache:
            cached = SAM3MultiRegionCollector._cache[cache_key]
            log.info(f"CACHE HIT - returning cached result for key={cache_key[:8]}")
            img_base64 = self.tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_base64]},
                "result": cached
            }

        log.info(f"CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse stored prompts
        try:
            raw_prompts = json.loads(multi_prompts_store) if multi_prompts_store.strip() else []
        except json.JSONDecodeError:
            raw_prompts = []

        img_height, img_width = image.shape[1], image.shape[2]
        log.info(f"Image dimensions: {img_width}x{img_height}")
        log.info(f"Processing {len(raw_prompts)} prompt regions")

        # Convert to normalized output format
        multi_prompts = []
        for idx, raw_prompt in enumerate(raw_prompts):
            prompt = {
                "id": idx,
                "positive_points": {"points": [], "labels": []},
                "negative_points": {"points": [], "labels": []},
                "positive_boxes": {"boxes": [], "labels": []},
                "negative_boxes": {"boxes": [], "labels": []},
            }

            # Normalize positive points
            for pt in raw_prompt.get("positive_points", []):
                norm_x = pt["x"] / img_width
                norm_y = pt["y"] / img_height
                prompt["positive_points"]["points"].append([norm_x, norm_y])
                prompt["positive_points"]["labels"].append(1)

            # Normalize negative points
            for pt in raw_prompt.get("negative_points", []):
                norm_x = pt["x"] / img_width
                norm_y = pt["y"] / img_height
                prompt["negative_points"]["points"].append([norm_x, norm_y])
                prompt["negative_points"]["labels"].append(0)

            # Normalize positive boxes (convert x1,y1,x2,y2 to center format)
            for box in raw_prompt.get("positive_boxes", []):
                x1_norm = box["x1"] / img_width
                y1_norm = box["y1"] / img_height
                x2_norm = box["x2"] / img_width
                y2_norm = box["y2"] / img_height
                cx = (x1_norm + x2_norm) / 2
                cy = (y1_norm + y2_norm) / 2
                w = x2_norm - x1_norm
                h = y2_norm - y1_norm
                prompt["positive_boxes"]["boxes"].append([cx, cy, w, h])
                prompt["positive_boxes"]["labels"].append(True)

            # Normalize negative boxes
            for box in raw_prompt.get("negative_boxes", []):
                x1_norm = box["x1"] / img_width
                y1_norm = box["y1"] / img_height
                x2_norm = box["x2"] / img_width
                y2_norm = box["y2"] / img_height
                cx = (x1_norm + x2_norm) / 2
                cy = (y1_norm + y2_norm) / 2
                w = x2_norm - x1_norm
                h = y2_norm - y1_norm
                prompt["negative_boxes"]["boxes"].append([cx, cy, w, h])
                prompt["negative_boxes"]["labels"].append(False)

            # Count items for logging
            pos_pts = len(prompt["positive_points"]["points"])
            neg_pts = len(prompt["negative_points"]["points"])
            pos_boxes = len(prompt["positive_boxes"]["boxes"])
            neg_boxes = len(prompt["negative_boxes"]["boxes"])
            log.info(f"  Prompt {idx}: {pos_pts} pos pts, {neg_pts} neg pts, {pos_boxes} pos boxes, {neg_boxes} neg boxes")

            # Only include prompts with content
            if (prompt["positive_points"]["points"] or
                prompt["negative_points"]["points"] or
                prompt["positive_boxes"]["boxes"] or
                prompt["negative_boxes"]["boxes"]):
                multi_prompts.append(prompt)

        log.info(f"Output: {len(multi_prompts)} non-empty prompts")

        # Cache and return
        result = (multi_prompts,)
        SAM3MultiRegionCollector._cache[cache_key] = result
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": result
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        img_array = tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64


class SAM3InteractiveCollector:
    """
    Interactive Collector with live segmentation preview.

    Same multi-region prompt UI as SAM3MultiRegionCollector, but also takes
    a SAM3 model and runs segmentation directly.  The widget has a "Run"
    button that calls a custom API route for instant mask overlay without
    having to queue the full workflow.
    """
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model from LoadSAM3Model node."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Image to segment. Draw points/boxes on the canvas, then click Run for a live mask preview."
                }),
                "multi_prompts_store": ("STRING", {"multiline": False, "default": "[]"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "SAM3_MULTI_PROMPTS")
    RETURN_NAMES = ("masks", "visualization", "multi_prompts")
    FUNCTION = "segment"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, sam3_model, image, multi_prompts_store, unique_id=None):
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(multi_prompts_store.encode())
        return h.hexdigest()

    # -- helpers reused by both segment() and the API route ----------------

    @staticmethod
    def _parse_raw_prompts(raw_prompts, img_w, img_h):
        """Normalize raw JS prompts (pixel coords) to model format."""
        multi_prompts = []
        for idx, raw in enumerate(raw_prompts):
            prompt = {
                "id": idx,
                "positive_points": {"points": [], "labels": []},
                "negative_points": {"points": [], "labels": []},
                "positive_boxes": {"boxes": [], "labels": []},
                "negative_boxes": {"boxes": [], "labels": []},
            }
            for pt in raw.get("positive_points", []):
                prompt["positive_points"]["points"].append([pt["x"] / img_w, pt["y"] / img_h])
                prompt["positive_points"]["labels"].append(1)
            for pt in raw.get("negative_points", []):
                prompt["negative_points"]["points"].append([pt["x"] / img_w, pt["y"] / img_h])
                prompt["negative_points"]["labels"].append(0)
            for box in raw.get("positive_boxes", []):
                x1n, y1n = box["x1"] / img_w, box["y1"] / img_h
                x2n, y2n = box["x2"] / img_w, box["y2"] / img_h
                prompt["positive_boxes"]["boxes"].append([
                    (x1n + x2n) / 2, (y1n + y2n) / 2, x2n - x1n, y2n - y1n
                ])
                prompt["positive_boxes"]["labels"].append(True)
            for box in raw.get("negative_boxes", []):
                x1n, y1n = box["x1"] / img_w, box["y1"] / img_h
                x2n, y2n = box["x2"] / img_w, box["y2"] / img_h
                prompt["negative_boxes"]["boxes"].append([
                    (x1n + x2n) / 2, (y1n + y2n) / 2, x2n - x1n, y2n - y1n
                ])
                prompt["negative_boxes"]["labels"].append(False)
            has_content = (prompt["positive_points"]["points"] or
                           prompt["negative_points"]["points"] or
                           prompt["positive_boxes"]["boxes"] or
                           prompt["negative_boxes"]["boxes"])
            if has_content:
                multi_prompts.append(prompt)
        return multi_prompts

    @staticmethod
    def _run_prompts(model, state, multi_prompts, img_w, img_h):
        """Run predict_inst for each prompt, return stacked masks + scores."""
        all_masks = []
        all_scores = []
        for prompt in multi_prompts:
            pts, labels = [], []
            for pt in prompt["positive_points"]["points"]:
                pts.append([pt[0] * img_w, pt[1] * img_h])
                labels.append(1)
            for pt in prompt["negative_points"]["points"]:
                pts.append([pt[0] * img_w, pt[1] * img_h])
                labels.append(0)
            box_array = None
            pos_boxes = prompt.get("positive_boxes", {}).get("boxes", [])
            if pos_boxes:
                cx, cy, w, h = pos_boxes[0]
                box_array = np.array([
                    (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                    (cx + w / 2) * img_w, (cy + h / 2) * img_h,
                ])
            point_coords = np.array(pts) if pts else None
            point_labels = np.array(labels) if labels else None
            if point_coords is None and box_array is None:
                continue
            masks_np, scores_np, _ = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                mask_input=None,
                multimask_output=True,
                normalize_coords=True,
            )
            best_idx = np.argmax(scores_np)
            all_masks.append(torch.from_numpy(masks_np[best_idx]).float())
            all_scores.append(scores_np[best_idx])
        return all_masks, all_scores

    # -- main execution (workflow queue) -----------------------------------

    def segment(self, sam3_model, image, multi_prompts_store, unique_id=None):
        import comfy.model_management
        comfy.model_management.load_models_gpu([sam3_model])
        pil_image = comfy_image_to_pil(image)
        img_w, img_h = pil_image.size

        processor = sam3_model.processor
        model = processor.model  # The actual nn.Module with predict_inst

        # Sync processor device after model load
        if hasattr(processor, 'sync_device_with_model'):
            processor.sync_device_with_model()

        state = processor.set_image(pil_image)

        # Cache for the API route
        _INTERACTIVE_CACHE[str(unique_id)] = {
            "sam3_model": sam3_model,
            "model": model,
            "processor": processor,
            "state": state,
            "pil_image": pil_image,
            "img_size": (img_w, img_h),
        }

        # Parse prompts
        try:
            raw_prompts = json.loads(multi_prompts_store) if multi_prompts_store.strip() else []
        except json.JSONDecodeError:
            raw_prompts = []
        multi_prompts = self._parse_raw_prompts(raw_prompts, img_w, img_h)

        # Run segmentation
        all_masks, all_scores = self._run_prompts(model, state, multi_prompts, img_w, img_h)

        if not all_masks:
            empty_mask = torch.zeros(1, img_h, img_w)
            vis_tensor = pil_to_comfy_image(pil_image)
            img_b64 = self._tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_b64]},
                "result": (empty_mask, vis_tensor, multi_prompts),
            }

        masks = torch.stack(all_masks, dim=0)
        scores = torch.tensor(all_scores)

        # Bounding boxes for visualization
        boxes_list = []
        for i in range(masks.shape[0]):
            coords = torch.where(masks[i] > 0)
            if len(coords[0]) > 0:
                boxes_list.append([coords[1].min().item(), coords[0].min().item(),
                                   coords[1].max().item(), coords[0].max().item()])
            else:
                boxes_list.append([0, 0, 0, 0])
        boxes = torch.tensor(boxes_list).float()

        comfy_masks = masks_to_comfy_mask(masks)
        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        img_b64 = self._tensor_to_base64(image)
        overlay_b64 = self._pil_to_base64(vis_image)

        return {
            "ui": {"bg_image": [img_b64], "overlay_image": [overlay_b64]},
            "result": (comfy_masks, vis_tensor, multi_prompts),
        }

    @staticmethod
    def _tensor_to_base64(tensor):
        arr = tensor[0].cpu().numpy()
        arr = (arr * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _pil_to_base64(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Custom API route for live interactive segmentation
# ---------------------------------------------------------------------------

def _run_segment_sync(cached, raw_prompts):
    """Blocking helper — called from the async route via run_in_executor."""
    sam3_model = cached["sam3_model"]  # ModelPatcher for GPU management
    model = cached["model"]            # processor.model with predict_inst
    state = cached["state"]
    pil_image = cached["pil_image"]
    img_w, img_h = cached["img_size"]

    import comfy.model_management
    comfy.model_management.load_models_gpu([sam3_model])

    multi_prompts = SAM3InteractiveCollector._parse_raw_prompts(raw_prompts, img_w, img_h)
    if not multi_prompts:
        return {"error": "No valid prompts", "num_masks": 0}

    all_masks, all_scores = SAM3InteractiveCollector._run_prompts(
        model, state, multi_prompts, img_w, img_h
    )
    if not all_masks:
        return {"error": "No masks generated", "num_masks": 0}

    masks = torch.stack(all_masks, dim=0)
    scores = torch.tensor(all_scores)

    boxes_list = []
    for i in range(masks.shape[0]):
        coords = torch.where(masks[i] > 0)
        if len(coords[0]) > 0:
            boxes_list.append([coords[1].min().item(), coords[0].min().item(),
                               coords[1].max().item(), coords[0].max().item()])
        else:
            boxes_list.append([0, 0, 0, 0])
    boxes = torch.tensor(boxes_list).float()

    vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
    buf = io.BytesIO()
    vis_image.save(buf, format="JPEG", quality=80)
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"overlay": overlay_b64, "num_masks": len(all_masks)}


def _run_segment_sync_one(cached, raw_prompt, prompt_name):
    """Run segmentation for a single named prompt. Thread-safe via _SEGMENT_LOCK."""
    log.info("Prompt '%s' dispatched", prompt_name)

    sam3_model = cached["sam3_model"]
    model = cached["model"]
    state = cached["state"]
    pil_image = cached["pil_image"]
    img_w, img_h = cached["img_size"]

    import comfy.model_management
    comfy.model_management.load_models_gpu([sam3_model])

    multi_prompts = SAM3InteractiveCollector._parse_raw_prompts([raw_prompt], img_w, img_h)
    if not multi_prompts:
        log.info("Prompt '%s' result ready (no valid points/boxes)", prompt_name)
        return {"error": "No valid prompt content", "num_masks": 0}

    with _SEGMENT_LOCK:
        all_masks, all_scores = SAM3InteractiveCollector._run_prompts(
            model, state, multi_prompts, img_w, img_h
        )

    log.info("Prompt '%s' result ready", prompt_name)

    if not all_masks:
        return {"error": "No masks generated", "num_masks": 0}

    return {"num_masks": len(all_masks)}


if _SERVER_AVAILABLE:
    @server.PromptServer.instance.routes.post("/sam3/interactive_segment_one")
    async def _interactive_segment_one_handler(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        node_id = str(body.get("node_id", ""))
        raw_prompt = body.get("prompt", {})
        prompt_name = str(body.get("prompt_name", "Prompt"))

        cached = _INTERACTIVE_CACHE.get(node_id)
        if not cached:
            return web.json_response(
                {"error": "Model not loaded. Queue the workflow first (Ctrl+Enter)."},
                status=400,
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _run_segment_sync_one, cached, raw_prompt, prompt_name
            )
            return web.json_response(result)
        except Exception as exc:
            log.exception("Interactive segmentation (single prompt '%s') failed", prompt_name)
            return web.json_response({"error": str(exc)}, status=500)

    @server.PromptServer.instance.routes.post("/sam3/interactive_segment")
    async def _interactive_segment_handler(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        node_id = str(body.get("node_id", ""))
        raw_prompts = body.get("prompts", [])

        cached = _INTERACTIVE_CACHE.get(node_id)
        if not cached:
            return web.json_response(
                {"error": "Model not loaded. Queue the workflow first (Ctrl+Enter)."},
                status=400,
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _run_segment_sync, cached, raw_prompts)
            return web.json_response(result)
        except Exception as exc:
            log.exception("Interactive segmentation failed")
            return web.json_response({"error": str(exc)}, status=500)

    @server.PromptServer.instance.routes.post("/sam3/get_frame")
    async def _get_frame_handler(request):
        """Return a single video frame as JPEG for the PointCollector slider."""
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400)

        node_id   = str(body.get("node_id", ""))
        frame_idx = int(body.get("frame_idx", 0))

        frames = _POINT_COLLECTOR_FRAMES.get(node_id)
        if frames is None:
            return web.Response(status=404)

        idx = max(0, min(frame_idx, frames.shape[0] - 1))
        frame_np = frames[idx].cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        pil_img  = Image.fromarray(frame_np)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return web.Response(body=buf.read(), content_type="image/jpeg")

    @server.PromptServer.instance.routes.get("/sam3/list_videos")
    async def _list_videos_handler(request):
        """List .mp4 files under ComfyUI's output directory for the file picker."""
        import os
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        video_files = []

        for root, dirs, files in os.walk(output_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in sorted(files):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                    full_path = os.path.join(root, fname)
                    rel_path  = os.path.relpath(full_path, output_dir)
                    video_files.append({
                        "path": full_path,
                        "label": rel_path,
                        "mtime": os.path.getmtime(full_path),
                    })

        # Sort newest first
        video_files.sort(key=lambda x: x["mtime"], reverse=True)
        return web.json_response(video_files)

    @server.PromptServer.instance.routes.post("/sam3/get_mask_frame")
    async def _get_mask_frame_handler(request):
        """Return a single mask frame as JPEG for overlay in PointCollector.
        Tensor cache (mask_frames input) takes priority over file path."""
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400)

        node_id   = str(body.get("node_id", ""))
        frame_idx = int(body.get("frame_idx", 0))

        # --- Try tensor cache first (from visualization output) ---
        tensors = _POINT_COLLECTOR_MASK_TENSORS.get(node_id)
        if tensors is not None:
            idx = max(0, min(frame_idx, tensors.shape[0] - 1))
            frame_np = tensors[idx].cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            pil_img  = Image.fromarray(frame_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return web.Response(body=buf.read(), content_type="image/jpeg")

        # --- Fallback: file path ---
        import cv2
        path = _POINT_COLLECTOR_MASK_PATHS.get(node_id)
        if not path:
            return web.Response(status=404)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return web.Response(status=404)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx   = max(0, min(frame_idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return web.Response(status=404)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return web.Response(body=buf.read(), content_type="image/jpeg")


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SAM3PointCollector": SAM3PointCollector,
    "SAM3AnimalPointCollector": SAM3AnimalPointCollector,
    "SAM3BBoxCollector": SAM3BBoxCollector,
    "SAM3MultiRegionCollector": SAM3MultiRegionCollector,
    "SAM3InteractiveCollector": SAM3InteractiveCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3PointCollector": "SAM3 Point Collector",
    "SAM3AnimalPointCollector": "SAM3 Animal Point Collector",
    "SAM3BBoxCollector": "SAM3 BBox Collector",
    "SAM3MultiRegionCollector": "SAM3 Multi-Region Collector",
    "SAM3InteractiveCollector": "SAM3 Interactive Collector",
}
