#!/usr/bin/env python3
"""
Convert Ultralytics results.csv to a TensorBoard event file.

Usage:
    python convert_results_csv_to_tensorboard.py --csv runs/exp/results.csv --out runs/exp/tb --run-name myrun

If pandas is installed it will be used, otherwise the stdlib csv module is used.
"""
import argparse
import os
import math

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

def read_csv(csv_path):
    if pd is not None:
        return pd.read_csv(csv_path)
    # fallback to csv module -> return list of dicts as minimal dataframe-like object
    import csv
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    # convert to simple dict-of-lists
    cols = reader.fieldnames
    data = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            data[c].append(r[c])
    # return as "pseudo-dataframe" (dict-of-lists)
    return data

def safe_float(x):
    # Try convert to float, else return NaN
    try:
        if x is None:
            return float('nan')
        if isinstance(x, float) or isinstance(x, int):
            return float(x)
        return float(str(x).strip())
    except Exception:
        return float('nan')

def to_tensorboard(csv_path, out_dir, run_name=None, tag_prefix=None):
    if SummaryWriter is None:
        raise RuntimeError("torch and tensorboard support not available. Install 'torch' and 'tensorboard'.")

    data = read_csv(csv_path)
    # If pandas, data is DataFrame. If fallback, dict-of-lists.
    if pd is not None and isinstance(data, pd.DataFrame):
        df = data
        cols = list(df.columns)
    else:
        df = None
        cols = list(data.keys())

    # Determine step column: prefer 'epoch' if present, else use row index
    step_col = None
    for c in ('epoch', 'Epoch', 'ep'):
        if c in cols:
            step_col = c
            break

    # Exclude columns we don't want to log as scalars
    exclude = {step_col, 'time', 'date', 'timestamp'} if step_col else {'time', 'date', 'timestamp'}
    scalar_cols = [c for c in cols if c not in exclude]

    # Prepare output dir
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir) if run_name is None else SummaryWriter(log_dir=os.path.join(out_dir, run_name))

    n_rows = len(df) if df is not None else len(next(iter(data.values())))
    for i in range(n_rows):
        step = None
        if step_col:
            if df is not None:
                val = df.iloc[i][step_col]
            else:
                val = data[step_col][i]
            # attempt to coerce to int
            try:
                step = int(float(val))
            except Exception:
                step = i
        else:
            step = i

        for col in scalar_cols:
            if df is not None:
                raw = df.iloc[i][col]
            else:
                raw = data[col][i]
            val = safe_float(raw)
            tag = f"{col}" if not tag_prefix else f"{tag_prefix}/{col}"
            # Only log finite numbers
            if math.isfinite(val):
                writer.add_scalar(tag, val, step)
    writer.flush()
    writer.close()
    print(f"Done. TensorBoard events written to {out_dir}")
    print(f"Run tensorboard: tensorboard --logdir {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ultralytics results.csv to TensorBoard events")
    parser.add_argument("--csv", default='/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect_d4q/minieye-driving-d4q-yolo11s_c2f_deconv-full_image-all/results.csv', help="Path to results.csv")
    parser.add_argument("--out", default='/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect_d4q/', help="Output directory for TensorBoard logs")
    parser.add_argument("--run-name", default='minieye-driving-d4q-yolo11s_c2f_deconv-full_image-all', help="Optional subdirectory name inside out for this run")
    parser.add_argument("--tag-prefix", default=None, help="Optional prefix for tags (e.g. 'train' or 'val')")
    args = parser.parse_args()
    to_tensorboard(args.csv, args.out, run_name=args.run_name, tag_prefix=args.tag_prefix)