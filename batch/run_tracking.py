"""
Batch SAM3 tracking runner — Step 3 of the fear-synchrony batch pipeline.

Reads batch/pipeline.csv, submits a SAM3 tracking job to ComfyUI for every
row with status=prompted, waits for completion, then extracts masks to CSV.

Usage:
    python batch/run_tracking.py [--csv PATH] [--dry-run] [--comfyui URL]

Prerequisites:
    - ComfyUI running at localhost:8188 (or --comfyui)
    - pipeline.csv populated by the ComfyUI annotation workflow (SAM3WritePipelineCSV)
"""

import argparse
import csv
import json
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR  = REPO_ROOT / "batch"
DEFAULT_CSV = BATCH_DIR / "pipeline.csv"
EXTRACT_SCRIPT = REPO_ROOT / "extract_masks_from_propagation.py"
PYTHON = sys.executable   # same env that runs this script

POLL_INTERVAL = 5   # seconds between history polls
POLL_TIMEOUT  = 3600  # max seconds to wait per job (1 hour)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

PIPELINE_HEADERS = [
    "video_id", "video_path", "status",
    "prompt_frame_idx", "mouse1_x", "mouse1_y", "mouse2_x", "mouse2_y",
    "output_pt_path", "masks_csv_path",
]


def load_pipeline(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def save_pipeline(csv_path: Path, rows: list[dict]):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PIPELINE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# ComfyUI workflow builder
# ---------------------------------------------------------------------------

def build_tracking_workflow(video_path: str, prompt_store_json: str,
                             video_id: str, output_dir: Path,
                             force: bool = False) -> dict:
    """
    Minimal node graph:
      1: LoadSAM3Model
      2: VHS_LoadVideoPath  (absolute path)
      3: SAM3TwoMouseTracking  (video_frames from node 2, prompt_store injected)
      4: SAM3Propagate  (sam3_model from 1, video_state from 3)
      5: SAM3SavePropagation  (masks/scores/video_state from 4)
    Passing an absolute path as filename_prefix causes os.path.join to ignore
    ComfyUI's output_dir, so the .pt file lands in output_dir directly.
    """
    prefix = str(output_dir / f"{video_id}_sam3_propa")
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
            "class_type": "VHS_LoadVideoPath",
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
            "class_type": "SAM3TwoMouseTracking",
            "inputs": {
                "video_frames": ["2", 0],
                "prompt_store": prompt_store_json,
                # Tiny random perturbation when force=True busts ComfyUI's node cache
                # without affecting tracking results (threshold stays effectively 0.3).
                "score_threshold": 0.3 + (random.random() * 1e-9 if force else 0),
            },
        },
        "4": {
            "class_type": "SAM3Propagate",
            "inputs": {
                "sam3_model": ["1", 0],
                "video_state": ["3", 0],
                "start_frame": 0,
                "end_frame": -1,
                "direction": "both",
            },
        },
        "5": {
            "class_type": "SAM3SavePropagation",
            "inputs": {
                "masks":           ["4", 0],
                "scores":          ["4", 1],
                "video_state":     ["4", 2],
                "filename_prefix": prefix,
            },
        },
    }


# ---------------------------------------------------------------------------
# Workflow template export
# ---------------------------------------------------------------------------

WORKFLOW_JSON = (
    Path(__file__).resolve().parent.parent
    / "comfyui-sam3-fearsync" / "workflows" / "batch_tracking.json"
)

PLACEHOLDER_PROMPT_STORE = json.dumps([
    {"frame_idx": 0, "obj_id": 1, "points": [[100.0, 80.0]], "labels": [1],
     "img_width": 304, "img_height": 236},
    {"frame_idx": 0, "obj_id": 2, "points": [[200.0, 150.0]], "labels": [1],
     "img_width": 304, "img_height": 236},
])


def save_workflow_template(out_path: Path = WORKFLOW_JSON):
    """Save batch_tracking.json with placeholder values for reference / ComfyUI API testing."""
    workflow = build_tracking_workflow(
        video_path="/path/to/video.mp4",
        prompt_store_json=PLACEHOLDER_PROMPT_STORE,
        video_id="example",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(workflow, f, indent=2)
    print(f"Saved workflow template → {out_path}")


# ---------------------------------------------------------------------------
# ComfyUI API helpers
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
    """Poll /history/{prompt_id} until the job is done. Return outputs dict."""
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
                        msgs = status.get("messages", [])
                        raise RuntimeError(f"ComfyUI job failed: {msgs}")
                    return entry.get("outputs", {})
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {prompt_id} did not finish within {POLL_TIMEOUT}s")


def get_save_path_from_outputs(outputs: dict, save_node_id: str = "5") -> str | None:
    """Extract the .pt file path from SAM3SavePropagation's output."""
    node_out = outputs.get(save_node_id, {})
    paths = node_out.get("save_path", [])
    if paths:
        return paths[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",      default=str(DEFAULT_CSV),
                        help="Path to pipeline.csv")
    parser.add_argument("--comfyui",  default="http://localhost:8188",
                        help="ComfyUI base URL")
    parser.add_argument("--video-id", default=None,
                        help="Re-run tracking for a specific video_id regardless of status")
    parser.add_argument("--force", action="store_true",
                        help="Force re-execution by busting ComfyUI node cache (auto-enabled with --video-id)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for .pt and masks CSV output (default: <csv_dir>/output)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print jobs without submitting")
    parser.add_argument("--save-workflow", action="store_true",
                        help=f"Save workflow template to {WORKFLOW_JSON} and exit")
    args = parser.parse_args()

    if args.save_workflow:
        save_workflow_template()
        return

    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent / "output"

    rows = load_pipeline(csv_path)

    if args.video_id:
        pending = [r for r in rows if r.get("video_id") == args.video_id]
        if not pending:
            print(f"video_id '{args.video_id}' not found in pipeline.csv.")
            return
    else:
        pending = [r for r in rows if r.get("status") == "prompted"]
        if not pending:
            print("No rows with status=prompted in pipeline.csv. Nothing to do.")
            return

    print(f"Found {len(pending)} video(s) to track.")
    print(f"Output directory: {output_dir}")

    for row in pending:
        vid_id    = row["video_id"]
        vid_path  = row["video_path"]
        m1x, m1y  = row["mouse1_x"], row["mouse1_y"]
        m2x, m2y  = row["mouse2_x"], row["mouse2_y"]
        frame_idx = row["prompt_frame_idx"]

        print(f"\n--- {vid_id}: {vid_path}")

        # Build prompt_store JSON from CSV columns
        try:
            fx = int(frame_idx)
            prompt_store = json.dumps([
                {
                    "frame_idx":  fx,
                    "obj_id":     1,
                    "points":     [[float(m1x), float(m1y)]],
                    "labels":     [1],
                    "img_width":  304,
                    "img_height": 236,
                },
                {
                    "frame_idx":  fx,
                    "obj_id":     2,
                    "points":     [[float(m2x), float(m2y)]],
                    "labels":     [1],
                    "img_width":  304,
                    "img_height": 236,
                },
            ])
        except (ValueError, TypeError) as e:
            print(f"  SKIP: invalid prompt data — {e}")
            continue

        force = args.force or bool(args.video_id)
        workflow = build_tracking_workflow(vid_path, prompt_store, vid_id, output_dir, force=force)

        if args.dry_run:
            print(f"  [dry-run] Would submit workflow for {vid_id}")
            print(f"  prompt_store: {prompt_store}")
            continue

        # Submit to ComfyUI
        print(f"  Submitting to ComfyUI…")
        try:
            prompt_id = submit_prompt(args.comfyui, workflow)
        except Exception as e:
            print(f"  ERROR submitting: {e}")
            continue
        print(f"  prompt_id={prompt_id} — waiting for completion…")

        # Wait
        try:
            outputs = wait_for_completion(args.comfyui, prompt_id)
        except Exception as e:
            print(f"  ERROR waiting: {e}")
            continue

        pt_path = get_save_path_from_outputs(outputs)
        if not pt_path:
            print(f"  WARNING: could not find save_path in outputs: {outputs}")
            continue

        # Rename to clean filename: {video_id}_sam3_segmentation.pt
        output_dir.mkdir(parents=True, exist_ok=True)
        final_pt = output_dir / f"{vid_id}_sam3_segmentation.pt"
        Path(pt_path).rename(final_pt)
        pt_path = str(final_pt)
        print(f"  .pt saved → {pt_path}")

        # Extract masks CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_csv = output_dir / f"{vid_id}_masks.csv"
        print(f"  Extracting masks → {masks_csv}")
        result = subprocess.run(
            [PYTHON, str(EXTRACT_SCRIPT), pt_path, str(masks_csv)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: extract_masks failed:\n{result.stderr}")
        else:
            print(result.stdout.strip())

        # Update CSV
        for r in rows:
            if r["video_id"] == vid_id:
                r["output_pt_path"] = pt_path
                r["masks_csv_path"] = str(masks_csv)
                r["status"] = "tracked"
                break
        save_pipeline(csv_path, rows)
        print(f"  pipeline.csv updated → status=tracked")

    print("\nDone.")


if __name__ == "__main__":
    main()
