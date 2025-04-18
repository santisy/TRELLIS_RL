#!/usr/bin/env python3
import argparse
import os
import shutil
import glob
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Merge per-step CSVs into metadata.csv with MultiIndex"
    )
    parser.add_argument(
        "--dir", "-d",
        default=".",
        help="Directory containing metadata.csv and step CSVs"
    )
    args = parser.parse_args()
    out_dir = args.dir

    # 1. Load existing metadata with a MultiIndex on (sha256, hdri_name)
    meta_path = os.path.join(out_dir, "metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Cannot find {meta_path}")
    meta = pd.read_csv(
        meta_path,
        index_col=["sha256", "hdri_name"]
    )

    # 2. Prepare archive folder
    merged_dir = os.path.join(out_dir, "merged_records")
    os.makedirs(merged_dir, exist_ok=True)

    # 3. Merge each step CSV
    for step_csv in glob.glob(os.path.join(out_dir, "*_*.csv")):
        name = os.path.basename(step_csv)
        if name == "metadata.csv":
            continue

        # 3a. Read with same MultiIndex
        df = pd.read_csv(
            step_csv,
            index_col=["sha256", "hdri_name"]
        )

        # 3b. Update in place
        meta.update(df)

        # 3c. Archive processed CSV
        shutil.move(step_csv, os.path.join(merged_dir, name))

    # 4. Write back the merged metadata
    meta.to_csv(meta_path)

if __name__ == "__main__":
    main()
