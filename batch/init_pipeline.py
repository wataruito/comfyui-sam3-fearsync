"""
Initialize pipeline.csv with video files found by a glob pattern.

If pipeline.csv already exists, shows the diff (new videos to add, missing
videos to delete) and asks for confirmation before making changes.

Usage:
    python batch/init_pipeline.py [GLOB] [--csv PATH] [--dry-run]

Examples:
    # Use defaults (video/*/*.mp4 relative to cwd, pipeline.csv in cwd)
    python batch/init_pipeline.py

    # Explicit glob pattern
    python batch/init_pipeline.py "/home/ubg/.../video/*/*.mp4"

    # Custom output path
    python batch/init_pipeline.py --csv /path/to/pipeline.csv
"""

import argparse
import csv
import glob
from pathlib import Path

FIELDNAMES = [
    "video_id",
    "video_path",
    "status",
    "prompt_frame_idx",
    "mouse1_x",
    "mouse1_y",
    "mouse2_x",
    "mouse2_y",
    "output_pt_path",
    "masks_csv_path",
]


def video_id(video_path: Path) -> str:
    """Generate video_id as {folder}_{stem}, e.g. 20230823_f337"""
    folder = video_path.parent.name
    return f"{folder}_{video_path.stem}"


def load_existing(csv_path: Path) -> dict:
    """Return dict of {video_id: row} for all rows in the CSV."""
    if not csv_path.exists():
        return {}
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            return {row["video_id"]: row for row in reader}
    except OSError as e:
        print(f"Error: Cannot open {csv_path} — {e}")
        print("Please close the file if it is open in another program and try again.")
        raise SystemExit(1)
    except csv.Error as e:
        print(f"Error: Failed to parse {csv_path}: {e}")
        raise SystemExit(1)


def find_videos(pattern: str) -> list[Path]:
    return sorted(Path(p).resolve() for p in glob.glob(pattern))


def ask(prompt: str) -> bool:
    """Ask a yes/no question. Returns True for yes."""
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def main():
    parser = argparse.ArgumentParser(description="Initialize pipeline.csv with video paths.")
    parser.add_argument(
        "glob",
        nargs="?",
        default="video/*/*.mp4",
        metavar="GLOB",
        help='Glob pattern for videos (default: "video/*/*.mp4" relative to cwd)',
    )
    parser.add_argument(
        "--csv",
        default=str(Path.cwd() / "pipeline.csv"),
        help="Output pipeline.csv path (default: ./pipeline.csv)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print diff without writing")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    existing = load_existing(csv_path)
    videos = find_videos(args.glob)
    found_ids = {video_id(v): v for v in videos}

    to_add = {vid: path for vid, path in found_ids.items() if vid not in existing}
    to_delete = {vid: row for vid, row in existing.items() if vid not in found_ids}

    if not to_add and not to_delete:
        print(f"pipeline.csv is up to date ({len(existing)} entries).")
        return

    # --- Show diff ---
    if to_add:
        print(f"\n[+] {len(to_add)} video(s) to add:")
        for vid, path in sorted(to_add.items()):
            print(f"    {vid:40s}  {path}")

    if to_delete:
        print(f"\n[-] {len(to_delete)} video(s) in CSV but not found on disk:")
        for vid, row in sorted(to_delete.items()):
            print(f"    {vid:40s}  {row['video_path']}  (status={row['status']})")

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return

    # --- Confirm add ---
    do_add = False
    if to_add:
        do_add = ask(f"\nAdd {len(to_add)} new entry(s) to {csv_path}?")

    # --- Confirm delete ---
    do_delete = False
    if to_delete:
        do_delete = ask(f"Delete {len(to_delete)} missing entry(s) from {csv_path}?")

    if not do_add and not do_delete:
        print("No changes made.")
        return

    # --- Apply changes ---
    # Rebuild rows: keep existing (minus deleted), append new
    kept_rows = [row for vid, row in existing.items() if not (do_delete and vid in to_delete)]

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in kept_rows:
                writer.writerow(row)
            if do_add:
                for vid, path in sorted(to_add.items()):
                    writer.writerow({
                        "video_id": vid,
                        "video_path": str(path),
                        "status": "new",
                        "prompt_frame_idx": "",
                        "mouse1_x": "",
                        "mouse1_y": "",
                        "mouse2_x": "",
                        "mouse2_y": "",
                        "output_pt_path": "",
                        "masks_csv_path": "",
                    })
    except OSError as e:
        print(f"Error: Cannot write to {csv_path} — {e}")
        print("Please close the file if it is open in another program and try again.")
        raise SystemExit(1)

    added = len(to_add) if do_add else 0
    deleted = len(to_delete) if do_delete else 0
    print(f"Done. Added {added}, deleted {deleted} entries. Total: {len(kept_rows) + added}")


if __name__ == "__main__":
    main()
