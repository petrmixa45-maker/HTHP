import os
os.system('cls' if os.name == 'nt' else 'clear')
from multiprocessing import Pool, cpu_count
from multiprocessing import Process, Queue
import CoolProp.CoolProp as cp
from CoolProp.CoolProp import AbstractState
# cesta k REFPROP (změň podle svého počítače)
os.environ["RPPREFIX"] = r"C:\Program Files\REFPROP"
import numpy as np
from itertools import combinations
import h5py

cesta1 = os.getcwd()
cesta = os.path.join(cesta1, "směsi", "kvaternární směsi", "data směsí")
os.makedirs(cesta, exist_ok=True)

# vstupní parametry
T_evap = 273.15 + 80    # teplota v K
T_cond = 273.15 + 130   # teplota v K
dT_SH = 5               # přehřátí v K
dT_SC = 5               # podchlazení v K
eta_comp = 0.75         # účinnost kompresoru
Q_out = 500000          # požadovaný výkon ve W

T1 = T_evap + dT_SH
T3 = T_cond - dT_SC

n = 20  # počet kroků pro výpočet
podily = np.round(np.linspace(0, 1, n + 1), 2)


def build_valid_slices(podil_values):
    valid_slices = []

    for i, podil4 in enumerate(podil_values):
        slice_points = []
        max_sum_jk = 1.0 - podil4

        for j, podil3 in enumerate(podil_values):
            remaining = round(max_sum_jk - podil3, 2)
            if remaining < 0:
                continue

            max_k = min(len(podil_values) - 1, int(round(remaining * n)))
            for k in range(max_k + 1):
                podil2 = podil_values[k]
                podil1 = round(1.0 - podil2 - podil3 - podil4, 2)
                if podil1 < 0:
                    continue
                slice_points.append((j, k, podil1, podil2, podil3, podil4))

        valid_slices.append((i, slice_points))

    return valid_slices


VALID_SLICES = build_valid_slices(podily)


def worker_mixture(args):
    latka1, latka2, latka3, latka4 = args

    AS = AbstractState("REFPROP", f"{latka1}&{latka2}&{latka3}&{latka4}")
    errors = []
    slices = []

    for i, slice_points in VALID_SLICES:
        n_points = len(slice_points)
        jk_indices = np.empty((n_points, 2), dtype=np.int16)
        podil_data = np.empty((n_points, 4), dtype=np.float64)
        stav_data = np.empty((n_points, 5, 5), dtype=np.float64)
        obeh_data = np.empty((n_points, 7), dtype=np.float64)
        valid_idx = 0

        for j, k, podil1, podil2, podil3, podil4 in slice_points:
            AS.set_mole_fractions([podil1, podil2, podil3, podil4])

            try:
                AS.update(cp.QT_INPUTS, 1, T_evap)
                P1 = AS.p()

                AS.specify_phase(cp.iphase_gas)
                AS.update(cp.PT_INPUTS, P1, T1)
                h1 = AS.hmass()
                s1 = AS.smass()
                ro1 = AS.rhomass()

                AS.unspecify_phase()
                AS.update(cp.QT_INPUTS, 0, T_cond)
                P3 = AS.p()

                AS.specify_phase(cp.iphase_liquid)
                AS.update(cp.PT_INPUTS, P3, T3)
                h3 = AS.hmass()
                s3 = AS.smass()
                ro3 = AS.rhomass()

                AS.unspecify_phase()
                s2s = s1
                P2s = P3
                AS.update(cp.PSmass_INPUTS, P2s, s2s)
                h2s = AS.hmass()
                T2s = AS.T()
                ro2s = AS.rhomass()

                h2 = h1 + (h2s - h1) / eta_comp
                P2 = P2s
                AS.update(cp.HmassP_INPUTS, h2, P2)
                T2 = AS.T()
                s2 = AS.smass()
                ro2 = AS.rhomass()

                h4 = h3
                P4 = P1
                AS.update(cp.HmassP_INPUTS, h4, P4)
                T4 = AS.T()
                s4 = AS.smass()
                ro4 = AS.rhomass()
            except Exception as exc:
                AS.unspecify_phase()
                errors.append(
                    (i, j, k, podil1, podil2, podil3, podil4, str(exc))
                )
                continue

            q_in = h1 - h4
            q_out = h2 - h3
            m_dot = Q_out / q_out
            w_cycle = q_out - q_in
            W_comp = w_cycle * m_dot
            COP = q_out / w_cycle
            VHC = q_out * ro1

            jk_indices[valid_idx] = (j, k)
            podil_data[valid_idx] = (podil1, podil2, podil3, podil4)
            stav_data[valid_idx] = (
                (P1, T1, h1, s1, ro1),
                (P2, T2, h2, s2, ro2),
                (P2s, T2s, h2s, s2s, ro2s),
                (P3, T3, h3, s3, ro3),
                (P4, T4, h4, s4, ro4),
            )
            obeh_data[valid_idx] = (q_in, q_out, m_dot, w_cycle, W_comp, COP, VHC)
            valid_idx += 1

        if valid_idx > 0:
            slices.append(
                {
                    "i": i,
                    "jk": jk_indices[:valid_idx],
                    "podil": podil_data[:valid_idx],
                    "stav": stav_data[:valid_idx],
                    "obeh": obeh_data[:valid_idx],
                }
            )

    return {
        "nazev": f"{latka1}&{latka2}&{latka3}&{latka4}",
        "smes": np.array([latka1, latka2, latka3, latka4], dtype="S"),
        "slices": slices,
        "errors": errors,
    }


def writer(queue, filename):
    with h5py.File(filename, "w") as f:
        while True:
            item = queue.get()
            if item is None:
                break

            kind = item["kind"]

            if kind == "start":
                grp = f.create_group(item["nazev"])
                grp.create_dataset("Smes", data=item["smes"])
                grp.create_group("slices")
                grp.create_group("errors")
                continue

            if kind == "slice":
                grp = f[item["nazev"]]["slices"].create_group(f"i_{item['i']:02d}")
                grp.create_dataset("jk_index", data=item["jk"])
                grp.create_dataset("Podil", data=item["podil"], compression="lzf")
                grp.create_dataset("Stav", data=item["stav"], compression="lzf")
                grp.create_dataset("Obeh", data=item["obeh"], compression="lzf")
                continue

            if kind == "errors":
                err_grp = f[item["nazev"]]["errors"]
                errors = item["errors"]

                if errors:
                    numeric = np.array(
                        [[e[0], e[1], e[2], e[3], e[4], e[5], e[6]] for e in errors],
                        dtype=np.float64,
                    )
                    messages = np.array([e[7].encode("utf-8") for e in errors], dtype="S512")
                else:
                    numeric = np.empty((0, 7), dtype=np.float64)
                    messages = np.empty((0,), dtype="S512")

                err_grp.create_dataset("context", data=numeric)
                err_grp.create_dataset("message", data=messages)


with open(os.path.join(cesta1, "směsi", "pure_fluids.txt"), "r") as f:
    latky = [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    queue = Queue(maxsize=20)
    writer_process = Process(target=writer, args=(queue, os.path.join(cesta, "BASIC_quat.h5")))
    writer_process.start()
    pool = Pool(cpu_count() - 1)

    mixture_args = list(combinations(latky, 4))
    results_iter = pool.imap_unordered(worker_mixture, mixture_args)

    for _ in range(len(mixture_args)):
        result = next(results_iter)

        queue.put({"kind": "start", "nazev": result["nazev"], "smes": result["smes"]})

        for slice_result in result["slices"]:
            queue.put(
                {
                    "kind": "slice",
                    "nazev": result["nazev"],
                    "i": slice_result["i"],
                    "jk": slice_result["jk"],
                    "podil": slice_result["podil"],
                    "stav": slice_result["stav"],
                    "obeh": slice_result["obeh"],
                }
            )

        queue.put({"kind": "errors", "nazev": result["nazev"], "errors": result["errors"]})

    pool.close()
    pool.join()
    queue.put(None)
    writer_process.join()
