"""
Reset a video's pipeline.csv entry back to status=new for re-annotation.

Usage:
    python batch/reset_video.py --csv /path/to/pipeline.csv --video-id VIDEO_ID [VIDEO_ID ...]
"""

import argparse
import csv
from pathlib import Path

PIPELINE_HEADERS = [
    "video_id", "video_path", "status",
    "prompt_frame_idx", "mouse1_x", "mouse1_y", "mouse2_x", "mouse2_y",
    "output_pt_path", "masks_csv_path", "masks_video_path",
]

RESET_FIELDS = [
    "status", "prompt_frame_idx",
    "mouse1_x", "mouse1_y", "mouse2_x", "mouse2_y",
    "output_pt_path", "masks_csv_path", "masks_video_path",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to pipeline.csv")
    parser.add_argument("--video-id", nargs="+", required=True,
                        help="video_id(s) to reset")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    target_ids = set(args.video_id)
    found = set()

    for r in rows:
        if r["video_id"] in target_ids:
            for field in RESET_FIELDS:
                r[field] = ""
            r["status"] = "new"
            found.add(r["video_id"])
            print(f"Reset: {r['video_id']}")

    not_found = target_ids - found
    if not_found:
        print(f"WARNING: video_id not found: {sorted(not_found)}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PIPELINE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    print("pipeline.csv updated.")


if __name__ == "__main__":
    main()
