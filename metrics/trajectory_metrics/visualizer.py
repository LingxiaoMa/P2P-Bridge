import argparse
import math
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize denoising trajectories from saved step xyz files.")
    parser.add_argument(
        "--steps_root",
        type=str,
        required=True,
        help="Directory that contains per-shape step folders, e.g. output_objects/.../steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for trajectory plots. Defaults to <steps_root>/../trajectory_viz",
    )
    parser.add_argument(
        "--points_per_shape",
        type=int,
        default=1,
        help="Number of tracked points to visualize for each shape.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every kth tracked point to reduce visual clutter.",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="max_disp",
        choices=["max_disp", "uniform"],
        help="How to choose points to track. 'max_disp' is best for path visualization.",
    )
    parser.add_argument(
        "--zoom_scale",
        type=float,
        default=10.0,
        help="Magnification factor for the local path view around the selected point.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI of the saved figures.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=24.0,
        help="Elevation angle for 3D view.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=36.0,
        help="Azimuth angle for 3D view.",
    )
    parser.add_argument(
        "--line_alpha",
        type=float,
        default=0.95,
        help="Alpha value for trajectory lines.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=2.4,
        help="Line width for trajectory lines.",
    )
    parser.add_argument(
        "--scatter_size",
        type=float,
        default=4.0,
        help="Point size for step markers.",
    )
    parser.add_argument(
        "--show_step_markers",
        action="store_true",
        help="Draw all intermediate step markers in addition to the trajectory lines.",
    )
    parser.add_argument(
        "--start_end_marker_size",
        type=float,
        default=28.0,
        help="Point size for start and end markers.",
    )
    return parser.parse_args()


def list_shape_dirs(steps_root: str) -> List[str]:
    shape_dirs = []
    for name in sorted(os.listdir(steps_root)):
        path = os.path.join(steps_root, name)
        if os.path.isdir(path):
            shape_dirs.append(path)
    return shape_dirs


def load_shape_steps(shape_dir: str) -> np.ndarray:
    step_files = [f for f in os.listdir(shape_dir) if f.endswith(".xyz")]
    if not step_files:
        raise FileNotFoundError(f"No .xyz step files found in {shape_dir}")

    def extract_step_id(filename: str) -> int:
        stem = os.path.splitext(filename)[0]
        return int(stem.split("_")[-1])

    step_files = sorted(step_files, key=extract_step_id)
    steps = [np.loadtxt(os.path.join(shape_dir, fn), dtype=np.float32) for fn in step_files]
    return np.stack(steps, axis=0)


def sample_start_indices(num_points: int, target_points: int) -> np.ndarray:
    if target_points >= num_points:
        return np.arange(num_points)
    return np.linspace(0, num_points - 1, target_points, dtype=int)


def track_points_by_nearest_neighbor(step_clouds: np.ndarray, start_indices: np.ndarray) -> np.ndarray:
    num_steps = step_clouds.shape[0]
    tracks = np.zeros((num_steps, len(start_indices), 3), dtype=np.float32)
    current_indices = start_indices.copy()
    tracks[0] = step_clouds[0, current_indices]

    for step_idx in range(1, num_steps):
        tree = cKDTree(step_clouds[step_idx])
        _, nn_indices = tree.query(tracks[step_idx - 1], k=1)
        current_indices = nn_indices.astype(np.int64)
        tracks[step_idx] = step_clouds[step_idx, current_indices]

    return tracks


def choose_start_indices(step_clouds: np.ndarray, target_points: int, selection_mode: str) -> np.ndarray:
    num_points = step_clouds.shape[1]
    if selection_mode == "uniform":
        return sample_start_indices(num_points, target_points)

    all_indices = np.arange(num_points, dtype=np.int64)
    all_tracks = track_points_by_nearest_neighbor(step_clouds, all_indices)
    displacements = np.linalg.norm(all_tracks[-1] - all_tracks[0], axis=1)
    topk = min(target_points, num_points)
    best_indices = np.argsort(displacements)[-topk:]
    return np.sort(best_indices)


def set_axes_equal(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    if radius < 1e-8:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def project_track_to_2d(traj: np.ndarray) -> np.ndarray:
    centered = traj - traj.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return centered[:, :2]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    return centered @ basis


def plot_shape_trajectories(
    ax,
    tracks: np.ndarray,
    title: str,
    args,
    elev: float,
    azim: float,
    line_alpha: float,
    line_width: float,
    scatter_size: float,
):
    cmap = plt.get_cmap("viridis", tracks.shape[1])
    for idx in range(tracks.shape[1]):
        traj = tracks[:, idx, :]
        color = cmap(idx)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=color,
            alpha=line_alpha,
            linewidth=line_width,
            marker="o" if args.show_step_markers else None,
            markersize=max(1.5, scatter_size * 0.5),
        )
        if args.show_step_markers:
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                c=np.arange(tracks.shape[0]),
                cmap="plasma",
                s=scatter_size,
                vmin=0,
                vmax=max(tracks.shape[0] - 1, 1),
                depthshade=False,
            )

        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            traj[0, 2],
            color="#1f77b4",
            s=args.start_end_marker_size,
            edgecolors="black",
            linewidths=0.5,
            depthshade=False,
        )
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            traj[-1, 2],
            color="#d62728",
            s=args.start_end_marker_size,
            edgecolors="black",
            linewidths=0.5,
            depthshade=False,
        )

    all_points = tracks.reshape(-1, 3)
    set_axes_equal(ax, all_points)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_single_track_focus(fig, shape_name: str, step_clouds: np.ndarray, track: np.ndarray, args):
    ax_global = fig.add_subplot(121, projection="3d")
    base_cloud = step_clouds[0]
    ax_global.scatter(
        base_cloud[:, 0],
        base_cloud[:, 1],
        base_cloud[:, 2],
        s=1.2,
        c="#d0d0d0",
        alpha=0.18,
        depthshade=False,
    )
    ax_global.plot(
        track[:, 0],
        track[:, 1],
        track[:, 2],
        color="#d62728",
        linewidth=max(2.8, args.line_width),
        alpha=1.0,
        marker="o",
        markersize=4.0,
    )
    ax_global.scatter(
        track[0, 0],
        track[0, 1],
        track[0, 2],
        color="#1f77b4",
        s=max(40.0, args.start_end_marker_size + 8.0),
        edgecolors="black",
        linewidths=0.6,
        depthshade=False,
    )
    ax_global.scatter(
        track[-1, 0],
        track[-1, 1],
        track[-1, 2],
        color="#d62728",
        s=max(40.0, args.start_end_marker_size + 8.0),
        edgecolors="black",
        linewidths=0.6,
        depthshade=False,
    )
    set_axes_equal(ax_global, base_cloud)
    ax_global.view_init(elev=args.elev, azim=args.azim)
    ax_global.set_title(f"{shape_name}: global", fontsize=10)
    ax_global.set_xlabel("x")
    ax_global.set_ylabel("y")
    ax_global.set_zlabel("z")

    ax_zoom = fig.add_subplot(122)
    start = track[0]
    magnified = start + args.zoom_scale * (track - start)
    traj2d = project_track_to_2d(magnified)
    ax_zoom.plot(
        traj2d[:, 0],
        traj2d[:, 1],
        color="#d62728",
        linewidth=max(3.2, args.line_width + 0.8),
        alpha=0.95,
    )
    for idx in range(traj2d.shape[0] - 1):
        delta = traj2d[idx + 1] - traj2d[idx]
        ax_zoom.arrow(
            traj2d[idx, 0],
            traj2d[idx, 1],
            delta[0],
            delta[1],
            length_includes_head=True,
            head_width=0.015 * max(np.ptp(traj2d[:, 0]), np.ptp(traj2d[:, 1]), 1.0),
            head_length=0.025 * max(np.ptp(traj2d[:, 0]), np.ptp(traj2d[:, 1]), 1.0),
            fc="#d62728",
            ec="#d62728",
            alpha=0.9,
        )
    ax_zoom.scatter(traj2d[0, 0], traj2d[0, 1], c="#1f77b4", s=90, edgecolors="black", linewidths=0.6, zorder=3)
    ax_zoom.scatter(traj2d[-1, 0], traj2d[-1, 1], c="#d62728", s=90, edgecolors="black", linewidths=0.6, zorder=3)
    for idx in range(traj2d.shape[0]):
        ax_zoom.text(traj2d[idx, 0], traj2d[idx, 1], f"t{idx}", fontsize=8, ha="left", va="bottom")
    ax_zoom.set_title(f"magnified path x{args.zoom_scale:.1f}", fontsize=10)
    ax_zoom.set_xlabel("principal axis 1")
    ax_zoom.set_ylabel("principal axis 2")
    ax_zoom.grid(alpha=0.25)
    ax_zoom.set_aspect("equal", adjustable="box")

    pad = 0.15 * max(np.ptp(traj2d[:, 0]), np.ptp(traj2d[:, 1]), 1.0)
    ax_zoom.set_xlim(traj2d[:, 0].min() - pad, traj2d[:, 0].max() + pad)
    ax_zoom.set_ylim(traj2d[:, 1].min() - pad, traj2d[:, 1].max() + pad)


def save_single_shape_plot(shape_name: str, step_clouds: np.ndarray, tracks: np.ndarray, output_dir: str, args):
    if tracks.shape[1] == 1:
        fig = plt.figure(figsize=(10.5, 4.8))
        plot_single_track_focus(fig, shape_name, step_clouds, tracks[:, 0, :], args)
    else:
        fig = plt.figure(figsize=(6.4, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        plot_shape_trajectories(
            ax=ax,
            tracks=tracks,
            title=shape_name,
            args=args,
            elev=args.elev,
            azim=args.azim,
            line_alpha=args.line_alpha,
            line_width=args.line_width,
            scatter_size=args.scatter_size,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{shape_name}_traj.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def save_overview_grid(shape_tracks: List[np.ndarray], shape_names: List[str], output_dir: str, args):
    num_shapes = len(shape_names)
    cols = min(4, num_shapes)
    rows = int(math.ceil(num_shapes / cols))
    fig = plt.figure(figsize=(4.8 * cols, 4.2 * rows))

    for idx, (shape_name, tracks) in enumerate(zip(shape_names, shape_tracks), start=1):
        if tracks.shape[1] == 1:
            ax = fig.add_subplot(rows, cols, idx)
            traj2d = project_track_to_2d(tracks[:, 0, :])
            ax.plot(traj2d[:, 0], traj2d[:, 1], color="#d62728", linewidth=2.2)
            ax.scatter(traj2d[0, 0], traj2d[0, 1], c="#1f77b4", s=24, zorder=3)
            ax.scatter(traj2d[-1, 0], traj2d[-1, 1], c="#d62728", s=24, zorder=3)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            ax.set_title(shape_name, fontsize=10)
        else:
            ax = fig.add_subplot(rows, cols, idx, projection="3d")
            plot_shape_trajectories(
                ax=ax,
                tracks=tracks,
                title=shape_name,
                args=args,
                elev=args.elev,
                azim=args.azim,
                line_alpha=args.line_alpha,
                line_width=args.line_width,
                scatter_size=max(4.0, args.scatter_size - 2.0),
            )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectory_overview.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    steps_root = os.path.abspath(args.steps_root)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(steps_root), "trajectory_viz")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    shape_dirs = list_shape_dirs(steps_root)
    if not shape_dirs:
        raise FileNotFoundError(f"No shape step folders found in {steps_root}")

    tracked_shapes = []
    tracked_names = []

    for shape_dir in shape_dirs:
        shape_name = os.path.basename(shape_dir)
        step_clouds = load_shape_steps(shape_dir)
        start_indices = choose_start_indices(step_clouds, args.points_per_shape, args.selection_mode)
        tracks = track_points_by_nearest_neighbor(step_clouds, start_indices)
        tracks = tracks[:, :: max(args.stride, 1), :]

        save_single_shape_plot(shape_name, step_clouds, tracks, output_dir, args)
        tracked_shapes.append(tracks)
        tracked_names.append(shape_name)

    save_overview_grid(tracked_shapes, tracked_names, output_dir, args)
    print(f"Saved trajectory visualizations to {output_dir}")


if __name__ == "__main__":
    main()
