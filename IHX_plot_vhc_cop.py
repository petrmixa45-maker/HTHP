import argparse
import os
os.system('cls' if os.name == 'nt' else 'clear')
import h5py
import matplotlib.pyplot as plt
import numpy as np


# Tento skript cte vysledky z HDF5 souboru vytvoreneho skriptem IHX_codex.py
# a vykresli zavislost COP na VHC pro kvaternarni smesi s vnitrnim vymenikem IHX.
#
# Bezny postup:
# 1. Spustit vypocet v IHX_codex.py, ktery vytvori IHX_quat.h5.
# 2. Spustit tento skript:
#       py IHX_plot_vhc_cop.py
# 3. Vystupni PNG se ulozi do slozky cesta_save.
#
# Vetsinu beznych nastaveni najdes hned nize v bloku "Zakladni prepinace".
# Stejna nastaveni lze docasne prepsat i z prikazove radky, napriklad:
#       py IHX_plot_vhc_cop.py --show
#       py IHX_plot_vhc_cop.py --extreme-bins 6
#       py IHX_plot_vhc_cop.py --mixture R1233ZDE
#
# Poznamka k instalaci:
# Interaktivni hover popisky vyzaduji balicek mplcursors. Pokud neni
# nainstalovany, graf se vykresli i ulozi, jen nebudou fungovat popisky po
# najeti mysi.

# Sloupce v datasetu "Obeh":
# (q_in, q_out, m_dot, w_cycle, W_comp, COP, VHC)
OBEH_COP_COL = 5
OBEH_VHC_COL = 6

# Sloupce v datasetu "Stav":
# kazdy bod ma 7 stavu a v kazdem stavu hodnoty (P, T, h, s, ro).
# Poradi stavu v IHX_codex.py:
# 0 = stav 1 pred IHX, 1 = stav 1ihx za IHX a pred kompresorem,
# 2 = stav 2 za kompresorem, 3 = stav 2s, 4 = stav 3,
# 5 = stav 3ihx, 6 = stav 4.
# STAV_S_COL proto ukazuje na entropii s.
STAV_S_COL = 3
STAV_COMP_IN_IDX = 1
STAV_COMP_OUT_IDX = 2

DEFAULT_INPUT_FILE = "IHX_quat.h5"
DEFAULT_OUTPUT_FILE = "IHX_quat_VHC_COP.png"

# Zakladni prepinace pro bezne spousteni skriptu.
show = False        # True = po ulozeni PNG otevrit interaktivni okno matplotlib.
hover = True        # True = pri najeti mysi ukazovat slozeni bodu, pokud je mplcursors.

# Nastaveni hustotniho obarveni bodu.
# gaussian_kde je pro velke soubory vypocetne narocne, proto se KDE uci jen
# na nahodnem vzorku bodu a potom se pouzije pro obarveni vsech bodu.
density_sample_size = 5000
density_eval_batch = 50000
density_cmap = "viridis"

# Citlivost popisku extremu:
# 0 = popsat jen absolutni maximum VHC a absolutni maximum COP.
# Vyssi cislo = rozdelit data do vice pasem a popsat vice lokalnich maxim obalky.
# Napriklad 4 az 8 je dobry rozsah pro hledani vyraznych "okraju" mraku bodu.
extreme_bins = 0

# Zakladni fyzikalni filtry. Hodnoty pod temito mezemi se vubec nevykresli.
min_vhc = 0.0       # VHC je v grafu prepoctene na MJ/m3.
min_cop = 0.0

# Cesty jsou psane stejnym stylem jako v IHX_codex.py.
# Pokud IHX_quat.h5 jeste lezi primo v pracovni slozce, skript ho umi najit
# jako zalozni variantu. Preferovane misto je ale cesta_read.
cesta1 = os.getcwd()
cesta_read = os.path.join(cesta1, "směsi", "kvaternární směsi", "data směsí")
cesta_save = os.path.join(cesta1, "směsi", "kvaternární směsi", "grafy směsí")


def decode_names(raw_names):
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]


def format_point_label(fluid_names, fractions):
    # Tvar popisku bodu v grafu:
    # latka1[podil1]&latka2[podil2]&latka3[podil3]&latka4[podil4]
    return "&".join(
        f"{fluid}[{fraction:.2f}]" for fluid, fraction in zip(fluid_names, fractions)
    )


def resolve_input_path(input_file, input_dir):
    if os.path.isabs(input_file):
        return input_file

    h5_path = os.path.join(input_dir, input_file)
    fallback_path = os.path.join(cesta1, input_file)

    if not os.path.exists(h5_path) and os.path.exists(fallback_path):
        print(f"Soubor v cesta_read nenalezen, pouzivam soubor z pracovni slozky: {fallback_path}")
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
    # Generator cte HDF5 postupne po smesich a po rezech i_00, i_01, ...
    # Diky tomu nemusi drzet cely HDF5 soubor v pameti naraz.
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
                delta_s_comp = (
                    stav[:, STAV_COMP_OUT_IDX, STAV_S_COL]
                    - stav[:, STAV_COMP_IN_IDX, STAV_S_COL]
                )

                # Filtr vykreslovanych bodu:
                # - COP a VHC musi byt konecne hodnoty.
                # - VHC a COP musi byt nad nastavenymi minimy.
                # - s2 - s1ihx musi byt kladne, jinak jde o fyzikalne podezrely bod.
                # - vsechny 4 podily musi byt > 0, aby slo skutecne o kvaternarni smes.
                valid = (
                    np.isfinite(cop)
                    & np.isfinite(vhc_mj_m3)
                    & np.isfinite(delta_s_comp)
                    & (vhc_mj_m3 >= min_vhc_limit)
                    & (cop >= min_cop_limit)
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
    # Vybere body pro staticke popisky v PNG.
    # Vzdy popise absolutni maximum VHC a absolutni maximum COP.
    # Pokud bins > 0, rozdeli data do kvantilovych pasem a prida lokalni maxima:
    # - maximum COP v pasmech podle VHC,
    # - maximum VHC v pasmech podle COP.
    # Minima se zamerne nepopisuji.
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
    # Volitelne interaktivni popisky pro prohlizeni bodu v matplotlib okne.
    # Pro ulozene PNG nema hover vliv.
    try:
        import mplcursors
    except ImportError:
        print("Balicek mplcursors neni nainstalovan, interaktivni popisky se nezapnou.")
        return

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(selection):
        selection.annotation.set_text(labels[selection.index])
        selection.annotation.get_bbox_patch().set(alpha=0.9)


def estimate_point_density(vhc, cop, sample_size=density_sample_size, eval_batch=density_eval_batch):
    # Hustota bodu pro barevne rozliseni mraku.
    # Osy se nejdriv prepoctou na bezrozmerne z-skore, aby KDE neovladla osa
    # s vetsim ciselny rozsahem.
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("Balicek scipy neni nainstalovan, hustotni obarveni se vypne.")
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
        print(f"Hustotni obarveni se nepodarilo spocitat: {exc}")
        return None

    density = np.empty(point_count, dtype=float)
    for start in range(0, point_count, eval_batch):
        stop = min(start + eval_batch, point_count)
        density[start:stop] = kde(normalized[:, start:stop])

    return density


def make_plot(data, output_path, show=show, hover=hover, extreme_bins=extreme_bins):
    # Hlavni vykresleni. Velikost a pruhlednost bodu upravuj zde.
    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)

    density = estimate_point_density(data["vhc"], data["cop"])
    if density is not None:
        # Ridke body se kresli prvni, huste oblasti navrch.
        plot_order = np.argsort(density)
        scatter = ax.scatter(
            data["vhc"][plot_order],
            data["cop"][plot_order],
            c=density[plot_order],
            cmap=density_cmap,
            s=7,
            alpha=0.1,
            linewidths=0,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Relativní hustota bodů [-]")
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

    ax.set_title("COP kvaternárních směsí v závislosti na VHC - IHX")
    ax.set_xlabel("VHC [MJ/m3]")
    ax.set_ylabel("COP [-]")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    point_count = len(data["vhc"])
    mixture_count = len(set(data["mixture_names"]))
    ax.text(
        0.01,
        0.99,
        f"{point_count:,} bodů | {mixture_count:,} směsí".replace(",", " "),
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
        print(f"Graf ulozen do: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    # Argumenty prikazove radky jsou uzitecne pro jednorazove experimenty.
    # Trvalejsi nastaveni je pohodlnejsi menit v hornim bloku promennych.
    parser = argparse.ArgumentParser(
        description="Vykresli graf VHC vs. COP z vysledku v IHX_quat.h5."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Cesta k HDF5 souboru s vysledky.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Cesta k vystupnimu obrazku.",
    )
    parser.add_argument(
        "--input-dir",
        default=cesta_read,
        help="Slozka, ze ktere se cte HDF5 soubor.",
    )
    parser.add_argument(
        "--output-dir",
        default=cesta_save,
        help="Slozka, do ktere se uklada vystupni graf.",
    )
    parser.add_argument(
        "--mixture",
        help="Volitelny filtr nazvu smesi, napriklad R1233ZDE.",
    )
    parser.add_argument(
        "--max-mixtures",
        type=int,
        help="Volitelny limit poctu smesi, uzitecne pro rychly nahled.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=show,
        help="Po vykresleni otevrit interaktivni okno matplotlib.",
    )
    parser.add_argument(
        "--hover",
        action=argparse.BooleanOptionalAction,
        default=hover,
        help="Zapnout popisky bodu pri najeti mysi, vyzaduje balicek mplcursors.",
    )
    parser.add_argument(
        "--extreme-bins",
        type=int,
        default=extreme_bins,
        help="Pocet pasem pro hledani lokalnich extremu v obou osach.",
    )
    parser.add_argument(
        "--min-vhc",
        type=float,
        default=min_vhc,
        help="Minimalni vykreslovana hodnota VHC v MJ/m3.",
    )
    parser.add_argument(
        "--min-cop",
        type=float,
        default=min_cop,
        help="Minimalni vykreslovana hodnota COP.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    h5_path = os.path.abspath(resolve_input_path(args.input, args.input_dir))
    output_path = resolve_output_path(args.output, args.output_dir)
    output_path = os.path.abspath(output_path) if output_path else None

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Soubor neexistuje: {h5_path}")

    data = load_plot_data(
        h5_path,
        args.mixture,
        args.max_mixtures,
        include_labels=True,
        min_vhc_limit=args.min_vhc,
        min_cop_limit=args.min_cop,
    )
    if data is None:
        raise RuntimeError("V souboru nebyla nalezena zadna vykreslitelna data.")

    make_plot(data, output_path, show=args.show, hover=args.hover, extreme_bins=args.extreme_bins)


if __name__ == "__main__":
    main()
