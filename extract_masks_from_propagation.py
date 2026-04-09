"""
Extract per-mouse centroid and area from SAM3SavePropagation .pt files.

Output CSV columns:
  frame, s1x, s1y, s1area, s2x, s2y, s2area

Where s1=mouse1(obj_id=1, blue), s2=mouse2(obj_id=2, red).
Missing mask → centroid and area are NaN/0.

Usage:
  python extract_masks_from_propagation.py <path/to/sam3_result.pt> [output.csv]
"""

import sys
import os
import numpy as np
import torch
import pandas as pd


def centroid(mask_2d):
    """Return (cx, cy) of a boolean 2-D mask, or (nan, nan) if empty."""
    ys, xs = np.where(mask_2d)
    if len(xs) == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def extract(pt_path, out_csv=None):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    masks = data["masks"]  # {frame_idx: ndarray(num_objs, H, W)}

    rows = []
    for fi in sorted(masks.keys()):
        m = masks[fi]
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        m = m.astype(bool)

        row = {"frame": fi}

        for obj_ch, prefix in enumerate(["s1", "s2"]):
            if m.shape[0] > obj_ch:
                mask_2d = m[obj_ch]
                cx, cy = centroid(mask_2d)
                area = int(mask_2d.sum())
            else:
                cx, cy, area = float("nan"), float("nan"), 0
            row[f"{prefix}x"] = cx
            row[f"{prefix}y"] = cy
            row[f"{prefix}area"] = area

        rows.append(row)

    df = pd.DataFrame(rows, columns=["frame", "s1x", "s1y", "s1area", "s2x", "s2y", "s2area"])

    if out_csv is None:
        base = os.path.splitext(os.path.basename(pt_path))[0]
        out_csv = os.path.join("analysis_output", f"{base}_masks.csv")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} frames → {out_csv}")

    # Quick summary
    for prefix, label in [("s1", "mouse1"), ("s2", "mouse2")]:
        valid = df[f"{prefix}area"] > 0
        print(f"  {label}: {valid.sum()}/{len(df)} frames with mask "
              f"(area mean={df.loc[valid, f'{prefix}area'].mean():.0f} px)")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pt_path = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else None
    extract(pt_path, out_csv)
