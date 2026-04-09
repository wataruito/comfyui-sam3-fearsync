"""
Batch SAM3 correction runner — Step 5 of the fear-synchrony batch pipeline.

Reads batch/pipeline.csv and batch/corrections.csv, submits a SAM3 correction
job to ComfyUI for every row with status=corrected, waits for completion, then
re-extracts masks to CSV.

Usage:
    python batch/run_corrections.py [--csv PATH] [--corrections PATH] [--dry-run] [--comfyui URL]

Prerequisites:
    - ComfyUI running at localhost:8188 (or --comfyui)
    - pipeline.csv with status=corrected rows
    - corrections.csv populated by the ComfyUI correction annotation workflow
"""

import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR  = REPO_ROOT / "batch"
DEFAULT_PIPELINE_CSV    = BATCH_DIR / "pipeline.csv"
DEFAULT_CORRECTIONS_CSV = BATCH_DIR / "corrections.csv"
ANALYSIS_DIR = REPO_ROOT / "analysis_output"
EXTRACT_SCRIPT = REPO_ROOT / "extract_masks_from_propagation.py"
PYTHON = sys.executable

POLL_INTERVAL = 5
POLL_TIMEOUT  = 3600


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

PIPELINE_HEADERS = [
    "video_id", "video_path", "status",
    "prompt_frame_idx", "mouse1_x", "mouse1_y", "mouse2_x", "mouse2_y",
    "output_pt_path", "masks_csv_path",
]


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_pipeline(csv_path: Path, rows: list[dict]):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PIPELINE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Correction store builder
# ---------------------------------------------------------------------------

def build_correction_store(correction_rows: list[dict], img_w: int = 304,
                            img_h: int = 236) -> str:
    """
    Convert flat corrections.csv rows for one video into the JSON format
    that SAM3FrameCorrector expects for correction_store.

    corrections.csv columns: video_id, frame_idx, mouse_id, point_type, x, y

    Output format (list of prompt entries, one per unique frame+mouse):
      [{"frame_idx": N, "obj_id": M, "points": [[x,y],...], "labels": [1/0,...],
        "img_width": W, "img_height": H}, ...]
    """
    # Group by (frame_idx, mouse_id)
    groups: dict[tuple, dict] = {}
    for row in correction_rows:
        key = (int(row["frame_idx"]), int(row["mouse_id"]))
        if key not in groups:
            groups[key] = {"frame_idx": key[0], "obj_id": key[1],
                           "points": [], "labels": [],
                           "img_width": img_w, "img_height": img_h}
        groups[key]["points"].append([float(row["x"]), float(row["y"])])
        groups[key]["labels"].append(1 if row["point_type"] == "pos" else 0)

    return json.dumps(list(groups.values()))


# ---------------------------------------------------------------------------
# ComfyUI workflow builder
# ---------------------------------------------------------------------------

def build_correction_workflow(video_path: str, pt_path: str,
                               correction_store_json: str,
                               video_id: str) -> dict:
    """
    Correction node graph:
      1: LoadSAM3Model
      2: VHS_LoadVideo
      3: SAM3LoadPropagation  (loads existing .pt)
      4: SAM3FrameCorrector   (applies corrections, needs video_frames for image predictor)
      5: SAM3SavePropagation  (saves corrected masks)
    """
    prefix = f"video/{video_id}/sam3_corrected"
    return {
        "1": {
            "class_type": "LoadSAM3Model",
            "inputs": {
                "precision": "auto",
                "attention": "auto",
                "compile": False,
            },
        },
        "2": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_path,
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "AnimateDiff",
            },
        },
        "3": {
            "class_type": "SAM3LoadPropagation",
            "inputs": {
                "save_path": pt_path,
            },
        },
        "4": {
            "class_type": "SAM3FrameCorrector",
            "inputs": {
                "sam3_model":       ["1", 0],
                "masks":            ["3", 0],
                "video_state":      ["3", 1],
                "correction_store": correction_store_json,
                "video_frames":     ["2", 0],
            },
        },
        "5": {
            "class_type": "SAM3SavePropagation",
            "inputs": {
                "masks":           ["4", 0],
                "video_state":     ["3", 1],
                "filename_prefix": prefix,
            },
        },
    }


# ---------------------------------------------------------------------------
# ComfyUI API helpers  (same pattern as run_tracking.py)
# ---------------------------------------------------------------------------

def submit_prompt(comfyui_url: str, workflow: dict) -> str:
    prompt_id = str(uuid.uuid4())
    resp = requests.post(
        f"{comfyui_url}/prompt",
        json={"prompt": workflow, "prompt_id": prompt_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("node_errors"):
        raise RuntimeError(f"ComfyUI node errors: {data['node_errors']}")
    return data["prompt_id"]


def wait_for_completion(comfyui_url: str, prompt_id: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = requests.get(f"{comfyui_url}/history/{prompt_id}", timeout=10)
        if resp.status_code == 200:
            history = resp.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") in ("success", "error"):
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI job failed: {status.get('messages')}")
                    return entry.get("outputs", {})
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {prompt_id} did not finish within {POLL_TIMEOUT}s")


def get_save_path_from_outputs(outputs: dict, save_node_id: str = "5") -> str | None:
    node_out = outputs.get(save_node_id, {})
    paths = node_out.get("save_path", [])
    return paths[0] if paths else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",         default=str(DEFAULT_PIPELINE_CSV))
    parser.add_argument("--corrections", default=str(DEFAULT_CORRECTIONS_CSV))
    parser.add_argument("--comfyui",     default="http://localhost:8188")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    pipeline_path    = Path(args.csv)
    corrections_path = Path(args.corrections)

    rows       = load_csv(pipeline_path)
    corr_rows  = load_csv(corrections_path)

    # Group correction rows by video_id
    corr_by_vid: dict[str, list[dict]] = defaultdict(list)
    for r in corr_rows:
        corr_by_vid[r["video_id"]].append(r)

    pending = [r for r in rows if r.get("status") == "corrected"]
    if not pending:
        print("No rows with status=corrected in pipeline.csv. Nothing to do.")
        return

    print(f"Found {len(pending)} video(s) to correct.")

    for row in pending:
        vid_id   = row["video_id"]
        vid_path = row["video_path"]
        pt_path  = row["output_pt_path"]

        print(f"\n--- {vid_id}")

        if not pt_path:
            print(f"  SKIP: no output_pt_path (run_tracking.py first)")
            continue

        vid_corr = corr_by_vid.get(vid_id, [])
        if not vid_corr:
            print(f"  SKIP: no corrections found in corrections.csv")
            continue

        correction_store = build_correction_store(vid_corr)
        print(f"  {len(vid_corr)} correction points across "
              f"{len(json.loads(correction_store))} frame/mouse entries")

        workflow = build_correction_workflow(vid_path, pt_path,
                                             correction_store, vid_id)

        if args.dry_run:
            print(f"  [dry-run] Would submit correction workflow")
            print(f"  correction_store: {correction_store}")
            continue

        print(f"  Submitting to ComfyUI…")
        try:
            prompt_id = submit_prompt(args.comfyui, workflow)
        except Exception as e:
            print(f"  ERROR submitting: {e}")
            continue
        print(f"  prompt_id={prompt_id} — waiting…")

        try:
            outputs = wait_for_completion(args.comfyui, prompt_id)
        except Exception as e:
            print(f"  ERROR waiting: {e}")
            continue

        new_pt_path = get_save_path_from_outputs(outputs)
        if not new_pt_path:
            print(f"  WARNING: could not find save_path in outputs: {outputs}")
            continue
        print(f"  corrected .pt → {new_pt_path}")

        # Re-extract masks
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        masks_csv = ANALYSIS_DIR / f"{vid_id}_masks.csv"
        print(f"  Re-extracting masks → {masks_csv}")
        result = subprocess.run(
            [PYTHON, str(EXTRACT_SCRIPT), new_pt_path, str(masks_csv)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: extract_masks failed:\n{result.stderr}")
        else:
            print(result.stdout.strip())

        # Update pipeline CSV
        for r in rows:
            if r["video_id"] == vid_id:
                r["output_pt_path"] = new_pt_path
                r["masks_csv_path"] = str(masks_csv)
                r["status"] = "done"
                break
        save_pipeline(pipeline_path, rows)
        print(f"  pipeline.csv updated → status=done")

    print("\nDone.")


if __name__ == "__main__":
    main()
