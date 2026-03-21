# Fear Synchrony

ComfyUI + SAM3-based pipeline for analyzing **synchronized freeze behavior** (fear synchrony) between pairs of mice during fear conditioning experiments.

**Publications:**
- Ito W et al. (2025) *Biol Psychiatry Glob Open Sci.*
- Ito W et al. (2023) *Biol Psychiatry.*

---

## Overview

This repository provides a ComfyUI + SAM3 workflow for tracking two mice independently in front-angled camera recordings, detecting freeze behavior, and evaluating synchrony between the two animals.

The `comfyui-sam3/` directory is a fork of [PozzettiAndrea/ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3), extended with custom nodes for two-mouse tracking (symlinked from `~/ComfyUI/custom_nodes/comfyui-sam3`).

---

## Environment

| Item | Details |
|------|---------|
| ComfyUI | `~/ComfyUI`, conda env: `sam3` |
| GPU | NVIDIA RTX 2080 Ti (11 GB VRAM) |
| CUDA | 12.8 driver, PyTorch 2.5.1+cu121 |
| SAM3 model | `~/ComfyUI/models/sam3/sam3.pt` (3.3 GB, fp32) |

**Launch ComfyUI:**
```bash
cd ~/ComfyUI && ~/miniconda3/envs/sam3/bin/python main.py --listen --force-fp16
```

**Source patch required:**
`~/sam3/sam3/model/sam3_image.py:146` — `img_batch[unique_ids]` → `img_batch[unique_ids.cpu()]`

---

## Data

| File | Description |
|------|-------------|
| `20250718/New folder/f402ab.mp4` | 304×236 px, 4 fps, 722 frames — black mouse + white cage |
| `20250718/New folder/f402cd.mp4` | Same specs — agouti mouse + white cage |
| `20250718/New folder/_f402ab_track_freeze.csv` | Ground truth: `frame, s1x, s1y, s2x, s2y, s1freeze, s2freeze` |
| `20250718/New folder/_f402cd_track_freeze.csv` | Same format for f402cd |

> **Note:** Ground-truth coordinates are approximately 3× video pixel coordinates (`COORD_SCALE = 3.0`).

---

## Two-Mouse Tracking Workflow

```
Load Video → SAM3PointCollector → (single_prompt) → SAM3TwoMouseTracking → SAM3Propagate → SAM3VideoOutput
                  ↑                                         ↑                                    ↓
            video_frames                             video_frames                    visualization → SaveVideo
            mask_video_path ←────────────────────────────────────────────────── (saved video path)
```

**Steps:**
1. Connect `video_frames` to `SAM3PointCollector`
2. Select a frame with the slider; set `animal_id` (1 or 2)
3. Click on the canvas to place a point prompt → Queue (Ctrl+Enter)
4. Canvas auto-clears → repeat for next frame / animal
5. Use **Clear List** in `SAM3TwoMouseTracking` to reset all prompts
6. Connect `visualization` output → **SaveVideo** → supply the saved path to `mask_video_path` for overlay display

### SAM3VideoOutput Ports

| Port | Content | Use |
|------|---------|-----|
| `masks` | B&W binary (MASK type) | Individual mask processing |
| `visualization` | Color overlay (obj0=blue, obj1=red) | Save with SaveVideo → `mask_video_path` |
| `frames` | Original video frames | Reference |

---

## Custom Node Reference

The following nodes were added on top of [PozzettiAndrea/ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3).

---

#### `SAM3PointCollector`
*Modified from the original `SAM3PointCollector`.* Extended with:
- `animal_id` input (1–32) that maps directly to SAM3 `obj_id`
- `single_prompt` output (`SAM3_SINGLE_PROMPT`) for use with `SAM3TwoMouseTracking`
- Video playback slider with frame scrubbing
- **Mask: ON/OFF** toggle — overlays the saved visualization video on the canvas
- File selector (`<select>` + 🔄) to pick a visualization video from `output/`

| Input | Type | Description |
|-------|------|-------------|
| `video_frames` | IMAGE | All frames; enables slider |
| `frame_idx` | INT | Current frame |
| `animal_id` | INT | 1–32, maps to SAM3 obj_id |
| `mask_video_path` | STRING | Saved visualization video for overlay |

**Output:** `positive_points, negative_points, single_prompt`

---

#### `SAM3TwoMouseTracking`
Accumulates point prompts from `SAM3PointCollector` across multiple queues and builds a `SAM3_VIDEO_STATE` for `SAM3Propagate`. Each queue appends the new prompt to an internal list. Includes **Clear List** (reset prompts) and **List Prompts** (print to terminal) buttons. The connected `SAM3PointCollector` canvas auto-clears after each queue.

| Input | Type | Description |
|-------|------|-------------|
| `video_frames` | IMAGE | |
| `new_prompt` | SAM3_SINGLE_PROMPT | From SAM3PointCollector; appended each run |
| `score_threshold` | FLOAT | |

**Output:** `SAM3_VIDEO_STATE`

---

#### `SAM3FrameCorrector`
Applies per-frame mask corrections without re-running full propagation. Calls `add_prompt` on specific frames and merges the result into the base masks. Useful for fixing tracking errors at frames where the two mice are in contact.

| Input | Type | Description |
|-------|------|-------------|
| `sam3_model` | SAM3_MODEL | |
| `masks` | SAM3_VIDEO_MASKS | Base masks from SAM3Propagate |
| `video_state` | SAM3_VIDEO_STATE | |
| `new_correction` | SAM3_SINGLE_PROMPT | Correction prompt |

**Output:** `corrected_masks (SAM3_VIDEO_MASKS)`

---

#### `SAM3SavePropagation` / `SAM3LoadPropagation`
Save and reload propagation results (`masks` + `video_state`) to/from disk under `output/sam3_propagation/*.pt`. Enables a correction-only workflow — run full propagation once, save, then iterate on corrections without re-propagating.

