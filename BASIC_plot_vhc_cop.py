import argparse
import os
os.system('cls' if os.name == 'nt' else 'clear')
import h5py
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np


# This script reads results from the HDF5 file created by BASIC_brute.py
# and plots COP against VHC for quaternary mixtures.
#
# Typical workflow:
# 1. Run BASIC_brute.py to create BASIC_quat.h5.
# 2. Run this script:
#       py BASIC_plot_vhc_cop.py
# 3. The output PNG is saved to cesta_save.
#
# Most common settings are in the "Basic switches" block below.
# The same settings can be overridden temporarily from the command line, for example:
#       py BASIC_plot_vhc_cop.py --show
#       py BASIC_plot_vhc_cop.py --extreme-bins 6
#       py BASIC_plot_vhc_cop.py --mixture R1233ZDE
#
# Installation note:
# Interactive hover labels require the mplcursors package. If it is not installed,
# the plot will still be drawn and saved, but hover labels will be disabled.

# Columns in the "Obeh" dataset:
# (q_in, q_out, m_dot, w_cycle, W_comp, COP, VHC)
OBEH_COP_COL = 5
OBEH_VHC_COL = 6

# Columns in the "Stav" dataset:
# each point has 5 states, each storing (P, T, h, s, ro).
# STAV_S_COL therefore points to entropy s.
STAV_S_COL = 3

DEFAULT_INPUT_FILE = "BASIC_quat.h5"
DEFAULT_OUTPUT_FILE = "BASIC_quat_VHC_COP.png"

# Basic switches for normal script runs.
show = False        # True = open an interactive matplotlib window after saving the PNG.
hover = True        # True = show point composition on mouse hover if mplcursors is available.

# Density-based point coloring.
# gaussian_kde is expensive for large files, so KDE is fitted only on a random
# sample and then evaluated for all plotted points.
density_sample_size = 5000
density_eval_batch = 50000
density_cmap = "viridis"

# Sensitivity of extreme-point labels:
# 0 = label only the absolute VHC maximum and absolute COP maximum.
# Higher values split data into more bands and label more local envelope maxima.
# Values around 4 to 8 are useful for finding visible point-cloud edges.
extreme_bins = 0

# Basic physical filters. Values below these limits are not plotted.
min_vhc = 0.0       # VHC is converted to MJ/m3 in the plot.
min_cop = 0.0

# Paths follow the same style as BASIC_brute.py.
# If BASIC_quat.h5 is still directly in the working directory, the script can
# use it as a fallback. The preferred location is cesta_read.
cesta1 = os.getcwd()
cesta_read = os.path.join(cesta1, "směsi", "kvaternární směsi", "data směsí")
cesta_save = os.path.join(cesta1, "směsi", "kvaternární směsi", "grafy směsí")


def decode_names(raw_names):
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]


def format_point_label(fluid_names, fractions):
    # Point label format:
    # fluid1[fraction1]&fluid2[fraction2]&fluid3[fraction3]&fluid4[fraction4]
    return "&".join(
        f"{fluid}[{fraction:.2f}]" for fluid, fraction in zip(fluid_names, fractions)
    )


def resolve_input_path(input_file, input_dir):
    if os.path.isabs(input_file):
        return input_file

    h5_path = os.path.join(input_dir, input_file)
    fallback_path = os.path.join(cesta1, input_file)

    if not os.path.exists(h5_path) and os.path.exists(fallback_path):
        print(f"File not found in cesta_read, using file from working directory: {fallback_path}")
        return fallback_path

    return h5_path


def resolve_output_path(output_file, output_dir):
    if not output_file:
        return None
    if os.path.isabs(output_file):
        return output_file
    return os.path.join(output_dir, output_file)


def iter_points(
    h5_path,
    mixture_filter=None,
    max_mixtures=None,
    include_labels=False,
    min_vhc_limit=min_vhc,
    min_cop_limit=min_cop,
):
    # The generator reads HDF5 gradually by mixture and by i_00, i_01, ...
    # slices, so the whole HDF5 file does not have to be held in memory.
    loaded_mixtures = 0

    with h5py.File(h5_path, "r") as h5:
        for mixture_name in h5.keys():
            if mixture_filter and mixture_filter.lower() not in mixture_name.lower():
                continue

            if max_mixtures is not None and loaded_mixtures >= max_mixtures:
                break

            mixture_group = h5[mixture_name]
            fluid_names = decode_names(mixture_group["Smes"][()])
            slices_group = mixture_group["slices"]

            for slice_name in sorted(slices_group.keys()):
                slice_group = slices_group[slice_name]
                obeh = slice_group["Obeh"][()]
                podil = slice_group["Podil"][()]
                stav = slice_group["Stav"][()]

                if obeh.size == 0:
                    continue

                cop = obeh[:, OBEH_COP_COL]
                vhc_mj_m3 = obeh[:, OBEH_VHC_COL] / 1_000_000.0
                delta_s_comp = stav[:, 1, STAV_S_COL] - stav[:, 0, STAV_S_COL]

                # Filter for plotted points:
                # - COP and VHC must be finite.
                # - VHC and COP must be above the configured limits.
                # - s2 - s1 must be positive; otherwise the point is suspicious.
                # - all 4 fractions must be > 0 to keep only true quaternary mixtures.
                valid = (
                    np.isfinite(cop)
                    & np.isfinite(vhc_mj_m3)
                    & np.isfinite(delta_s_comp)
                    & (vhc_mj_m3 >= min_vhc_limit)
                    & (cop >= min_cop_limit)
                    & (cop <= 5.4)
                    & (delta_s_comp > 0.0)
                    & np.all(podil > 0.0, axis=1)
                )
                if not np.any(valid):
                    continue

                labels = None
                if include_labels:
                    labels = [
                        format_point_label(fluid_names, fractions)
                        for fractions in podil[valid]
                    ]

                yield {
                    "mixture_name": mixture_name,
                    "fluid_names": fluid_names,
                    "vhc": vhc_mj_m3[valid],
                    "cop": cop[valid],
                    "podil": podil[valid],
                    "labels": labels,
                }

            loaded_mixtures += 1


def load_plot_data(
    h5_path,
    mixture_filter=None,
    max_mixtures=None,
    include_labels=False,
    min_vhc_limit=min_vhc,
    min_cop_limit=min_cop,
):
    vhc_parts = []
    cop_parts = []
    labels = []
    mixture_names = []

    for points in iter_points(
        h5_path,
        mixture_filter,
        max_mixtures,
        include_labels=include_labels,
        min_vhc_limit=min_vhc_limit,
        min_cop_limit=min_cop_limit,
    ):
        vhc = points["vhc"]
        cop = points["cop"]
        podil = points["podil"]
        mixture_name = points["mixture_name"]

        vhc_parts.append(vhc)
        cop_parts.append(cop)
        mixture_names.append(mixture_name)

        if include_labels:
            labels.extend(points["labels"])

    if not vhc_parts:
        return None

    return {
        "vhc": np.concatenate(vhc_parts),
        "cop": np.concatenate(cop_parts),
        "labels": labels if include_labels else None,
        "mixture_names": mixture_names,
    }


def choose_extreme_indices(vhc, cop, bins=16):
    # Select points for static labels in the PNG.
    # Always labels the absolute VHC maximum and absolute COP maximum.
    # If bins > 0, splits data into quantile bands and adds local maxima:
    # - maximum COP in VHC bands,
    # - maximum VHC in COP bands.
    # Minima are intentionally not labeled.
    selected = {
        #int(np.argmax(vhc)),
        #int(np.argmax(cop)),
    }

    def add_axis_maxima(axis_values, target_values):
        edges = np.unique(np.quantile(axis_values, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 2:
            return

        for left, right in zip(edges[:-1], edges[1:]):
            if right == edges[-1]:
                mask = (axis_values >= left) & (axis_values <= right)
            else:
                mask = (axis_values >= left) & (axis_values < right)

            indices = np.flatnonzero(mask)
            if len(indices) == 0:
                continue

            selected.add(int(indices[np.argmax(target_values[indices])]))

    add_axis_maxima(vhc, cop)
    add_axis_maxima(cop, vhc)

    return sorted(selected)


def annotate_extreme_points(ax, data, bins):
    labels = data["labels"]
    if not labels:
        return

    extreme_indices = choose_extreme_indices(data["vhc"], data["cop"], bins=bins)
    for order, idx in enumerate(extreme_indices):
        x = data["vhc"][idx]
        y = data["cop"][idx]
        offset_x = 8 if order % 2 == 0 else -8
        offset_y = 8 if (order // 2) % 2 == 0 else -12
        ha = "left" if offset_x > 0 else "right"

        ax.scatter([x], [y], s=24, c="black", alpha=0.9, linewidths=0, zorder=4)
        ax.annotate(
            labels[idx],
            xy=(x, y),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "0.55", "alpha": 0.78, "pad": 1.2},
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.45},
            zorder=5,
        )


def add_hover_labels(scatter, labels):
    # Optional interactive labels for inspecting points in the matplotlib window.
    # Hover labels do not affect the saved PNG.
    try:
        import mplcursors
    except ImportError:
        print("Package mplcursors is not installed, interactive hover labels are disabled.")
        return

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(selection):
        selection.annotation.set_text(labels[selection.index])
        selection.annotation.get_bbox_patch().set(alpha=0.9)


def estimate_point_density(vhc, cop, sample_size=density_sample_size, eval_batch=density_eval_batch):
    # Point density for coloring the point cloud.
    # Axes are first converted to dimensionless z-scores so that KDE is not
    # dominated by the axis with the larger numerical range.
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("Package scipy is not installed, density coloring is disabled.")
        return None

    values = np.vstack((vhc, cop)).astype(float)
    center = np.mean(values, axis=1, keepdims=True)
    scale = np.std(values, axis=1, keepdims=True)
    scale[scale == 0.0] = 1.0
    normalized = (values - center) / scale

    point_count = normalized.shape[1]
    if point_count == 0:
        return None

    if point_count > sample_size:
        rng = np.random.default_rng(0)
        fit_indices = rng.choice(point_count, size=sample_size, replace=False)
        fit_values = normalized[:, fit_indices]
    else:
        fit_values = normalized

    try:
        kde = gaussian_kde(fit_values)
    except Exception as exc:
        print(f"Could not compute density coloring: {exc}")
        return None

    density = np.empty(point_count, dtype=float)
    for start in range(0, point_count, eval_batch):
        stop = min(start + eval_batch, point_count)
        density[start:stop] = kde(normalized[:, start:stop])

    return density


def make_plot(data, output_path, show=show, hover=hover, extreme_bins=extreme_bins):
    # Main plotting section. Adjust point size and opacity here.
    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)

    density = estimate_point_density(data["vhc"], data["cop"])
    if density is not None:
        # Sparse points are drawn first, dense regions on top.
        plot_order = np.argsort(density)
        density_norm = Normalize(vmin=np.min(density), vmax=np.max(density))
        scatter = ax.scatter(
            data["vhc"][plot_order],
            data["cop"][plot_order],
            c=density[plot_order],
            cmap=density_cmap,
            norm=density_norm,
            s=7,
            alpha=0.1,
            linewidths=0,
        )
        colorbar_mappable = ScalarMappable(norm=density_norm, cmap=density_cmap)
        colorbar_mappable.set_array([])
        cbar = fig.colorbar(colorbar_mappable, ax=ax)
        cbar.set_label("Relative point density [-]")
        hover_labels = [data["labels"][idx] for idx in plot_order] if data["labels"] else None
    else:
        scatter = ax.scatter(
            data["vhc"],
            data["cop"],
            color="tab:blue",
            s=7,
            alpha=0.1,
            linewidths=0,
        )
        hover_labels = data["labels"]

    ax.set_title("COP of Quaternary Mixtures vs. VHC - BASIC")
    ax.set_xlabel("VHC [MJ/m3]")
    ax.set_ylabel("COP [-]")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    point_count = len(data["vhc"])
    mixture_count = len(set(data["mixture_names"]))
    ax.text(
        0.01,
        0.99,
        f"{point_count:,} points | {mixture_count:,} mixtures".replace(",", " "),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    fig.tight_layout()

    annotate_extreme_points(ax, data, bins=extreme_bins)

    if hover and hover_labels:
        add_hover_labels(scatter, hover_labels)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    # Command-line arguments are useful for one-off experiments.
    # For persistent settings, it is usually easier to edit the variables above.
    parser = argparse.ArgumentParser(
        description="Plot VHC vs. COP from results stored in BASIC_quat.h5."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Path to the HDF5 result file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the output image.",
    )
    parser.add_argument(
        "--input-dir",
        default=cesta_read,
        help="Directory used for reading the HDF5 file.",
    )
    parser.add_argument(
        "--output-dir",
        default=cesta_save,
        help="Directory used for saving the output plot.",
    )
    parser.add_argument(
        "--mixture",
        help="Optional mixture-name filter, for example R1233ZDE.",
    )
    parser.add_argument(
        "--max-mixtures",
        type=int,
        help="Optional mixture-count limit, useful for quick previews.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=show,
        help="Open an interactive matplotlib window after plotting.",
    )
    parser.add_argument(
        "--hover",
        action=argparse.BooleanOptionalAction,
        default=hover,
        help="Enable point labels on mouse hover; requires mplcursors.",
    )
    parser.add_argument(
        "--extreme-bins",
        type=int,
        default=extreme_bins,
        help="Number of bands used for finding local envelope maxima on both axes.",
    )
    parser.add_argument(
        "--min-vhc",
        type=float,
        default=min_vhc,
        help="Minimum plotted VHC value in MJ/m3.",
    )
    parser.add_argument(
        "--min-cop",
        type=float,
        default=min_cop,
        help="Minimum plotted COP value.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    h5_path = os.path.abspath(resolve_input_path(args.input, args.input_dir))
    output_path = resolve_output_path(args.output, args.output_dir)
    output_path = os.path.abspath(output_path) if output_path else None

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File does not exist: {h5_path}")

    data = load_plot_data(
        h5_path,
        args.mixture,
        args.max_mixtures,
        include_labels=True,
        min_vhc_limit=args.min_vhc,
        min_cop_limit=args.min_cop,
    )
    if data is None:
        raise RuntimeError("No plottable data was found in the file.")

    make_plot(data, output_path, show=args.show, hover=args.hover, extreme_bins=args.extreme_bins)


if __name__ == "__main__":
    main()
