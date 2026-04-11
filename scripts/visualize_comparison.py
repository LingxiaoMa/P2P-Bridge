"""
Visualize the model comparison CSV produced by merge_results.py.

Generates grouped bar charts for each metric (cd_sph, p2f, plr, dc),
faceted by resolution, with noise level on the x-axis and model as groups.

Usage:
    python scripts/visualize_comparison.py \
        --csv output_comparison/comparison.csv \
        --out_dir output_comparison/plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


METRICS = {
    "cd_sph": "Chamfer Distance (unit sphere) ↓",
    "p2f": "Point-to-Face Distance ↓",
    "plr": "Path Length Ratio ↑",
    "dc": "Direction Consistency ↑",
}

MODEL_COLORS = {
    "baseline": "#4C72B0",
    "dit": "#DD8452",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="output_comparison/comparison.csv")
    parser.add_argument("--out_dir", type=str, default="output_comparison/plots")
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, metric: str, out_dir: str):
    resolutions = sorted(df["resolution"].unique())
    models = sorted(df["model"].unique())
    noises = sorted(df["noise"].unique())

    n_res = len(resolutions)
    fig, axes = plt.subplots(1, n_res, figsize=(6 * n_res, 5), sharey=False)
    if n_res == 1:
        axes = [axes]

    bar_width = 0.35
    x = np.arange(len(noises))

    for ax, res in zip(axes, resolutions):
        res_df = df[df["resolution"] == res]
        for i, model in enumerate(models):
            model_df = res_df[res_df["model"] == model].sort_values("noise")
            values = [
                model_df[model_df["noise"] == n][metric].values[0]
                if len(model_df[model_df["noise"] == n]) > 0 else float("nan")
                for n in noises
            ]
            offset = (i - (len(models) - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                values,
                bar_width,
                label=model,
                color=MODEL_COLORS.get(model, None),
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            # value labels on top
            for bar, v in zip(bars, values):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{v:.4f}",
                        ha="center", va="bottom", fontsize=7, rotation=45,
                    )

        ax.set_title(f"Resolution {res:,}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"σ={n}" for n in noises])
        ax.set_xlabel("Noise Level")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Model")

    fig.suptitle(METRICS[metric], fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{metric}_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary_table(df: pd.DataFrame, out_dir: str):
    """Print a formatted summary table and save as PNG."""
    pivot_rows = []
    for _, row in df.iterrows():
        pivot_rows.append({
            "Model": row["model"],
            "Res": int(row["resolution"]),
            "Noise": row["noise"],
            "CD-sph": f"{row['cd_sph']:.4f}",
            "P2F": f"{row['p2f']:.4f}",
            "PLR": f"{row['plr']:.4f}",
            "DC": f"{row['dc']:.4f}",
        })
    tbl = pd.DataFrame(pivot_rows)
    tbl = tbl.sort_values(["Res", "Noise", "Model"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(3, len(tbl) * 0.4 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # header styling
    for j in range(len(tbl.columns)):
        table[(0, j)].set_facecolor("#2d2d2d")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # row striping
    for i in range(1, len(tbl) + 1):
        color = "#f0f4ff" if i % 2 == 0 else "white"
        for j in range(len(tbl.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title("Full Comparison Table", fontsize=12, fontweight="bold", pad=10)
    out_path = os.path.join(out_dir, "summary_table.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    for metric in METRICS:
        plot_metric(df, metric, args.out_dir)

    plot_summary_table(df, args.out_dir)
    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
