"""
Generate mask overlay videos for already-tracked videos that are missing masks_video_path.

Usage:
    python batch/generate_mask_videos.py --csv /path/to/pipeline.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tracking import generate_mask_video, PIPELINE_HEADERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to pipeline.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    targets = [
        r for r in rows
        if r["status"] == "tracked"
        and r.get("output_pt_path")
        and not r.get("masks_video_path")
    ]

    if not targets:
        print("No tracked videos missing masks_video_path. Nothing to do.")
        return

    print(f"Found {len(targets)} video(s) to process.")

    for r in targets:
        pt = r["output_pt_path"]
        vid = r["video_path"]
        out = Path(pt).parent / f"{r['video_id']}_masks.mp4"
        print(f"\n{r['video_id']}")
        print(f"  pt:  {pt}")
        print(f"  vid: {vid}")
        print(f"  out: {out}")
        try:
            generate_mask_video(pt, vid, str(out))
            r["masks_video_path"] = str(out)
            print(f"  OK")
        except Exception as e:
            print(f"  ERROR: {e}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PIPELINE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print("\npipeline.csv updated.")


if __name__ == "__main__":
    main()
