"""
Find problematic frames in masks CSV files.

Detects:
  1. Missing mask  -- s1area==0 or s2area==0
  2. Position jump -- centroid moves more than --jump-thresh pixels between frames
  3. ID swap       -- distance(s1→s2_prev) + distance(s2→s1_prev) < same-id distances

Usage:
    python batch/find_bad_frames.py --csv /path/to/pipeline.csv
    python batch/find_bad_frames.py --masks /path/to/video_masks.csv
"""

import argparse
import csv
import math
from pathlib import Path


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_bad_frames(masks_csv: Path, jump_thresh: float = 60.0) -> list[dict]:
    """
    Returns a list of dicts:
      {"frame": int, "reason": str, "detail": str}
    """
    rows = []
    with open(masks_csv, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "frame": int(r["frame"]),
                    "s1x": float(r["s1x"]) if r["s1x"] not in ("", "nan") else float("nan"),
                    "s1y": float(r["s1y"]) if r["s1y"] not in ("", "nan") else float("nan"),
                    "s1area": float(r["s1area"]),
                    "s2x": float(r["s2x"]) if r["s2x"] not in ("", "nan") else float("nan"),
                    "s2y": float(r["s2y"]) if r["s2y"] not in ("", "nan") else float("nan"),
                    "s2area": float(r["s2area"]),
                })
            except (ValueError, KeyError):
                continue

    bad = []

    for i, r in enumerate(rows):
        frame = r["frame"]

        # 1. Missing mask
        if r["s1area"] == 0:
            bad.append({"frame": frame, "reason": "missing_mask", "detail": "s1area=0"})
        if r["s2area"] == 0:
            bad.append({"frame": frame, "reason": "missing_mask", "detail": "s2area=0"})

        if i == 0:
            continue
        prev = rows[i - 1]

        # Skip if either frame has NaN
        if any(math.isnan(v) for v in [r["s1x"], r["s1y"], r["s2x"], r["s2y"],
                                        prev["s1x"], prev["s1y"], prev["s2x"], prev["s2y"]]):
            continue

        # 2. Position jump
        d1 = dist(r["s1x"], r["s1y"], prev["s1x"], prev["s1y"])
        d2 = dist(r["s2x"], r["s2y"], prev["s2x"], prev["s2y"])
        if d1 > jump_thresh:
            bad.append({"frame": frame, "reason": "jump",
                        "detail": f"s1 moved {d1:.1f}px from frame {prev['frame']}"})
        if d2 > jump_thresh:
            bad.append({"frame": frame, "reason": "jump",
                        "detail": f"s2 moved {d2:.1f}px from frame {prev['frame']}"})

        # 3. ID swap — swapped assignment has lower total distance
        d_same  = d1 + d2
        d_swap  = (dist(r["s1x"], r["s1y"], prev["s2x"], prev["s2y"]) +
                   dist(r["s2x"], r["s2y"], prev["s1x"], prev["s1y"]))
        if d_swap < d_same - 5:   # 5px hysteresis
            bad.append({"frame": frame, "reason": "id_swap",
                        "detail": f"swap cost {d_swap:.1f} < same cost {d_same:.1f}"})

    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",   help="pipeline.csv — process all tracked videos")
    group.add_argument("--masks", help="Single masks CSV file")
    parser.add_argument("--jump-thresh", type=float, default=60.0,
                        help="Pixel distance threshold for jump detection (default: 60)")
    parser.add_argument("--summary", action="store_true",
                        help="Print only summary counts, not individual frames")
    args = parser.parse_args()

    if args.masks:
        targets = [("(single)", Path(args.masks))]
    else:
        pipeline_csv = Path(args.csv)
        targets = []
        with open(pipeline_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "tracked" and row.get("masks_csv_path"):
                    targets.append((row["video_id"], Path(row["masks_csv_path"])))

    if not targets:
        print("No tracked videos with masks_csv_path found.")
        return

    for video_id, masks_csv in targets:
        if not masks_csv.exists():
            print(f"{video_id}: masks CSV not found — {masks_csv}")
            continue

        bad = find_bad_frames(masks_csv, jump_thresh=args.jump_thresh)

        missing = [b for b in bad if b["reason"] == "missing_mask"]
        jumps   = [b for b in bad if b["reason"] == "jump"]
        swaps   = [b for b in bad if b["reason"] == "id_swap"]
        missing_frames = sorted(set(b["frame"] for b in missing))

        print(f"\n=== {video_id} ===")
        print(f"  missing_mask: {len(missing_frames)} frames  |  jump: {len(jumps)}  |  id_swap: {len(swaps)}")

        if not args.summary:
            if missing_frames:
                print(f"  MISSING  frames: {missing_frames}")
            if jumps:
                for b in jumps:
                    print(f"  JUMP     frame {b['frame']:4d}: {b['detail']}")
            if swaps:
                for b in swaps:
                    print(f"  ID_SWAP  frame {b['frame']:4d}: {b['detail']}")

        if not bad:
            print("  All frames OK.")

    print()


if __name__ == "__main__":
    main()
