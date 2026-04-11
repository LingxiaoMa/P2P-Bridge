"""
Merge evaluation CSVs from multiple models into a single comparison CSV.

Expected directory structure:
    output_root/{model}/PUNet/P2P-Bridge_ema_steps_10_{res}_{noise}/Summary_PUNet.csv
    output_root/{model}/PUNet/P2P-Bridge_ema_steps_10_{res}_{noise}/Summary_PUNet_traj.csv

Output columns: model, resolution, noise, cd_sph, p2f, plr, dc
"""

import argparse
import os
import glob

import pandas as pd


RESOLUTIONS = [10000, 50000]
NOISES = [0.01, 0.02, 0.03]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="output_comparison")
    parser.add_argument("--models", nargs="+", default=["baseline", "dit"])
    parser.add_argument("--dataset", type=str, default="PUNet")
    parser.add_argument("--save_title", type=str, default="P2P-Bridge_ema_steps_10")
    parser.add_argument("--out_csv", type=str, default="output_comparison/comparison.csv")
    return parser.parse_args()


def read_summary(path: str) -> dict:
    """Read Summary_{dataset}.csv and return mean cd_sph and p2f as floats."""
    df = pd.read_csv(path, index_col=0)
    # The CSV has one row per experiment name; grab the last row (most recent run)
    row = df.iloc[-1]
    return {
        "cd_sph": float(row.get("cd_sph(mean)", float("nan"))),
        "p2f": float(row.get("p2f(mean)", float("nan"))),
    }


def read_traj_summary(path: str) -> dict:
    """Read Summary_{dataset}_traj.csv and return mean PLR and DC."""
    df = pd.read_csv(path, index_col=0)
    mean = df.mean(axis=0)
    return {
        "plr": float(mean.get("path_length_ratio", float("nan"))),
        "dc": float(mean.get("direction_consistency", float("nan"))),
    }


def main():
    args = parse_args()
    rows = []

    for model in args.models:
        for res in RESOLUTIONS:
            for noise in NOISES:
                subdir = f"{args.save_title}_{res}_{noise}"
                base = os.path.join(args.output_root, model, args.dataset, subdir)

                summary_path = os.path.join(base, f"Summary_{args.dataset}.csv")
                traj_path = os.path.join(base, f"Summary_{args.dataset}_traj.csv")

                row = {"model": model, "resolution": res, "noise": noise}

                if os.path.exists(summary_path):
                    row.update(read_summary(summary_path))
                else:
                    print(f"[WARN] Missing: {summary_path}")
                    row.update({"cd_sph": float("nan"), "p2f": float("nan")})

                if os.path.exists(traj_path):
                    row.update(read_traj_summary(traj_path))
                else:
                    print(f"[WARN] Missing: {traj_path}")
                    row.update({"plr": float("nan"), "dc": float("nan")})

                rows.append(row)

    df = pd.DataFrame(rows, columns=["model", "resolution", "noise", "cd_sph", "p2f", "plr", "dc"])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"Saved comparison CSV to {args.out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
