"""Plot per-block average loss curves from TTO sidecar JSON files.

Each sidecar JSON (produced by `inference_tto_gaussian.py`) maps
`gen_video -> block_index -> step_index -> epoch -> {loss_name: float}`.

This tool averages those losses across videos and across optimization
epochs, then writes one PNG per diffusion timestep plus one averaged-across-
timesteps PNG. Each PNG is a small grid of subplots, one per loss
component, with x = temporal block and y = average loss. Multiple sidecar
JSONs can be passed and are overlaid on the same axes — convenient for
comparing an optimized run against a `measure_only` (un-optimized) baseline.

Example:

    python tools/plot_per_block_losses.py \\
        --losses path/to/optimized_run.json --label optimized \\
        --losses path/to/baseline_run.json --label baseline \\
        --output_dir plots/lvbench_compare/

Outputs (under `--output_dir`):

    step_0.png, step_1.png, step_2.png, step_3.png  (one per diffusion timestep)
    averaged.png                                    (averaged across timesteps)
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# Default loss components plotted (and their on-figure ordering). Components
# absent from the JSON are silently skipped, so this list is also safe to use
# with measure-only runs that have only a subset of these.
DEFAULT_COMPONENTS: list[str] = [
    "loss_total",
    "loss_gaussian",
    "loss_reg",
    "loss_mean",
    "loss_var",
    "loss_skew",
    "loss_kurt",
    "loss_sfm",
    "loss_skew_low",
    "loss_kurt_low",
]


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def _aggregate(
    json_data: dict[str, Any],
) -> tuple[
    dict[int, dict[int, dict[str, float]]],
    dict[int, dict[str, float]],
]:
    """Average loss values over (videos × epochs).

    Returns:
        per_step[step_idx][block_idx][component] = mean across (videos × epochs).
        averaged[block_idx][component] = mean across (videos × steps × epochs).
    """
    per_step_samples: defaultdict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    aggregated_samples: defaultdict = defaultdict(lambda: defaultdict(list))

    for _video, video_losses in json_data.items():
        for block_str, block_dict in video_losses.items():
            block_idx = int(block_str)
            for step_str, step_dict in block_dict.items():
                step_idx = int(step_str)
                for _epoch, loss_components in step_dict.items():
                    for name, val in loss_components.items():
                        if not isinstance(val, (int, float)):
                            continue
                        per_step_samples[step_idx][block_idx][name].append(val)
                        aggregated_samples[block_idx][name].append(val)

    per_step = {
        step: {
            block: {n: float(np.mean(v)) for n, v in losses.items()}
            for block, losses in blocks.items()
        }
        for step, blocks in per_step_samples.items()
    }
    averaged = {
        block: {n: float(np.mean(v)) for n, v in losses.items()}
        for block, losses in aggregated_samples.items()
    }
    return per_step, averaged


def _curve(
    per_block: dict[int, dict[str, float]],
    component: str,
) -> tuple[list[int], list[float]]:
    blocks = sorted(per_block.keys())
    values = [per_block[b].get(component, np.nan) for b in blocks]
    return blocks, values


def _present_components(
    runs: dict[str, dict[int, dict[str, float]]],
    requested: list[str],
) -> list[str]:
    """Filter requested components to those that show up in at least one run."""
    seen: set[str] = set()
    for per_block in runs.values():
        for block_data in per_block.values():
            seen.update(block_data.keys())
    return [c for c in requested if c in seen]


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _grid_shape(n: int) -> tuple[int, int]:
    """Pick a roughly square subplot grid that fits n components."""
    if n <= 1:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _render_grid(
    runs: dict[str, dict[int, dict[str, float]]],
    components: list[str],
    title: str,
    output_path: pathlib.Path,
    log_scale: bool,
) -> None:
    components = _present_components(runs, components)
    if not components:
        print(f"  [skip] {output_path.name}: no requested components present in input")
        return

    rows, cols = _grid_shape(len(components))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False
    )
    flat_axes = axes.flatten()

    cmap = plt.get_cmap("tab10")
    run_colors: dict[str, Any] = {label: cmap(i % 10) for i, label in enumerate(runs)}

    for ci, comp in enumerate(components):
        ax = flat_axes[ci]
        any_data = False
        for label, per_block in runs.items():
            blocks, values = _curve(per_block, comp)
            if not blocks:
                continue
            ax.plot(
                blocks, values,
                color=run_colors[label],
                marker=".",
                markersize=4,
                linewidth=1.4,
                label=label,
            )
            any_data = True
        ax.set_title(comp, fontsize=10)
        ax.set_xlabel("block", fontsize=9)
        ax.set_ylabel("avg loss", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if log_scale and any_data:
            # Only switch to log if every plotted point is strictly positive.
            ymin = np.nanmin([
                np.nanmin(_curve(pb, comp)[1] or [np.nan])
                for pb in runs.values()
            ])
            if np.isfinite(ymin) and ymin > 0:
                ax.set_yscale("log")

    # Hide unused axes (when the grid is bigger than n components).
    for ax in flat_axes[len(components):]:
        ax.axis("off")

    # Single shared legend at the figure level.
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.99),
            ncol=min(len(handles), 4), fontsize=10, frameon=False,
        )

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--losses", action="append", required=True, metavar="PATH",
        help="Path to a TTO sidecar JSON file. Repeat to overlay multiple runs.",
    )
    parser.add_argument(
        "--label", action="append", default=[], metavar="LABEL",
        help="Label for the corresponding --losses file (in the same order). "
             "If omitted, the file stem is used.",
    )
    parser.add_argument(
        "--output_dir", required=True, metavar="DIR",
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--components", nargs="*", default=None, metavar="NAME",
        help=f"Loss components to include. Default: {DEFAULT_COMPONENTS}.",
    )
    parser.add_argument(
        "--log_scale", action="store_true",
        help="Use a log y-axis where all plotted values are strictly positive.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    losses_paths: list[pathlib.Path] = [pathlib.Path(p) for p in args.losses]
    labels: list[str] = list(args.label)
    while len(labels) < len(losses_paths):
        labels.append(losses_paths[len(labels)].stem)

    runs_per_step: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
    runs_averaged: dict[str, dict[int, dict[str, float]]] = {}
    for label, path in zip(labels, losses_paths):
        with path.open("r") as f:
            data = json.load(f)
        per_step, averaged = _aggregate(data)
        runs_per_step[label] = per_step
        runs_averaged[label] = averaged
        n_videos = len(data)
        n_steps = len(per_step)
        n_blocks = max((len(b) for b in per_step.values()), default=0)
        print(f"loaded {label}: videos={n_videos}, steps={n_steps}, blocks={n_blocks}")

    components = list(args.components) if args.components else DEFAULT_COMPONENTS

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_steps = sorted({s for ps in runs_per_step.values() for s in ps})
    for step_idx in all_steps:
        runs = {label: runs_per_step[label].get(step_idx, {}) for label in runs_per_step}
        _render_grid(
            runs=runs,
            components=components,
            title=f"Per-block loss — diffusion timestep index {step_idx}",
            output_path=output_dir / f"step_{step_idx}.png",
            log_scale=args.log_scale,
        )

    _render_grid(
        runs=runs_averaged,
        components=components,
        title="Per-block loss — averaged across diffusion timesteps",
        output_path=output_dir / "averaged.png",
        log_scale=args.log_scale,
    )

    print(f"plots written to {output_dir}")


if __name__ == "__main__":
    main()
