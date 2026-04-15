import os
os.system('cls' if os.name == 'nt' else 'clear')
from multiprocessing import Pool, cpu_count
from multiprocessing import Process, Queue
import CoolProp.CoolProp as cp
from CoolProp.CoolProp import AbstractState
from time import perf_counter
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
dT_IHX = 20             # teplotní rozdíl ve vnitřním výměníku v K

T1 = T_evap + dT_SH
T3 = T_cond - dT_SC

n = 20  # počet kroků pro výpočet
podily = np.round(np.linspace(0, 1, n + 1), 2)
PROFILE_ENABLED = True
PROFILE_TXT = os.path.join(cesta, "IHX_quat_profile.txt")


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


def empty_profile_stats():
    return {
        "worker_total": 0.0,
        "abstract_state_init": 0.0,
        "set_mole_fractions": 0.0,
        "state1_qt": 0.0,
        "state1_pt": 0.0,
        "state1ihx_pt": 0.0,
        "state3_qt": 0.0,
        "state3_pt": 0.0,
        "state3ihx_hp": 0.0,
        "state2s_ps": 0.0,
        "state2_hp": 0.0,
        "state4_hp": 0.0,
        "property_reads": 0.0,
        "cycle_calc": 0.0,
        "result_store": 0.0,
        "worker_points_total": 0,
        "worker_points_valid": 0,
        "worker_errors": 0,
        "main_total": 0.0,
        "main_queue_put": 0.0,
        "main_wait_results": 0.0,
        "main_collect_errors": 0.0,
        "main_mixtures": 0,
        "writer_total": 0.0,
        "writer_start_group": 0.0,
        "writer_slice_write": 0.0,
        "writer_errors_write": 0.0,
        "writer_items_start": 0,
        "writer_items_slice": 0,
        "writer_items_errors": 0,
    }


def merge_profile_stats(target, source):
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def format_profile_report(stats):
    worker_total = stats["worker_total"]
    writer_total = stats["writer_total"]
    main_total = stats["main_total"]
    lines = [
        "Profiling summary for IHX_multiprocess.py",
        f"PROFILE_ENABLED = {PROFILE_ENABLED}",
        "",
        "Counts:",
        f"  mixtures processed: {stats['main_mixtures']}",
        f"  worker points total: {stats['worker_points_total']}",
        f"  worker points valid: {stats['worker_points_valid']}",
        f"  worker errors: {stats['worker_errors']}",
        f"  writer start items: {stats['writer_items_start']}",
        f"  writer slice items: {stats['writer_items_slice']}",
        f"  writer error items: {stats['writer_items_errors']}",
        "",
        "Main process time [s]:",
        f"  total: {main_total:.6f}",
        f"  waiting for worker results: {stats['main_wait_results']:.6f}",
        f"  queue.put to writer: {stats['main_queue_put']:.6f}",
        f"  error aggregation: {stats['main_collect_errors']:.6f}",
        "",
        "Worker aggregated time [s]:",
        f"  total: {worker_total:.6f}",
        f"  AbstractState init: {stats['abstract_state_init']:.6f}",
        f"  set_mole_fractions: {stats['set_mole_fractions']:.6f}",
        f"  state1 QT: {stats['state1_qt']:.6f}",
        f"  state1 PT: {stats['state1_pt']:.6f}",
        f"  state1ihx PT: {stats['state1ihx_pt']:.6f}",
        f"  state3 QT: {stats['state3_qt']:.6f}",
        f"  state3 PT: {stats['state3_pt']:.6f}",
        f"  state3ihx HP: {stats['state3ihx_hp']:.6f}",
        f"  state2s PS: {stats['state2s_ps']:.6f}",
        f"  state2 HP: {stats['state2_hp']:.6f}",
        f"  state4 HP: {stats['state4_hp']:.6f}",
        f"  property reads: {stats['property_reads']:.6f}",
        f"  cycle calc: {stats['cycle_calc']:.6f}",
        f"  result store: {stats['result_store']:.6f}",
        "",
        "Writer process time [s]:",
        f"  total: {writer_total:.6f}",
        f"  start-group writes: {stats['writer_start_group']:.6f}",
        f"  slice writes: {stats['writer_slice_write']:.6f}",
        f"  error writes: {stats['writer_errors_write']:.6f}",
        "",
        "Relative shares:",
        f"  main queue.put / main total: {((stats['main_queue_put'] / main_total) * 100) if main_total else 0:.2f} %",
        f"  main wait / main total: {((stats['main_wait_results'] / main_total) * 100) if main_total else 0:.2f} %",
        f"  writer slice / writer total: {((stats['writer_slice_write'] / writer_total) * 100) if writer_total else 0:.2f} %",
        f"  state2s PS / worker total: {((stats['state2s_ps'] / worker_total) * 100) if worker_total else 0:.2f} %",
        f"  state2 HP / worker total: {((stats['state2_hp'] / worker_total) * 100) if worker_total else 0:.2f} %",
        f"  state4 HP / worker total: {((stats['state4_hp'] / worker_total) * 100) if worker_total else 0:.2f} %",
        f"  state1ihx PT / worker total: {((stats['state1ihx_pt'] / worker_total) * 100) if worker_total else 0:.2f} %",
        f"  state3ihx HP / worker total: {((stats['state3ihx_hp'] / worker_total) * 100) if worker_total else 0:.2f} %",
    ]
    return "\n".join(lines) + "\n"


def worker_mixture(args):
    latka1, latka2, latka3, latka4 = args

    profile = empty_profile_stats() if PROFILE_ENABLED else None
    worker_start = perf_counter() if PROFILE_ENABLED else None
    init_start = perf_counter() if PROFILE_ENABLED else None
    AS = AbstractState("REFPROP", f"{latka1}&{latka2}&{latka3}&{latka4}")
    if PROFILE_ENABLED:
        profile["abstract_state_init"] += perf_counter() - init_start

    errors = []
    slices = []

    for i, slice_points in VALID_SLICES:
        n_points = len(slice_points)
        jk_indices = np.empty((n_points, 2), dtype=np.int16)
        podil_data = np.empty((n_points, 4), dtype=np.float64)
        stav_data = np.empty((n_points, 7, 5), dtype=np.float64)
        obeh_data = np.empty((n_points, 7), dtype=np.float64)
        valid_idx = 0

        for j, k, podil1, podil2, podil3, podil4 in slice_points:
            if PROFILE_ENABLED:
                profile["worker_points_total"] += 1
                t0 = perf_counter()
            AS.set_mole_fractions([podil1, podil2, podil3, podil4])
            if PROFILE_ENABLED:
                profile["set_mole_fractions"] += perf_counter() - t0

            try:
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.QT_INPUTS, 1, T_evap)
                if PROFILE_ENABLED:
                    profile["state1_qt"] += perf_counter() - t0
                P1 = AS.p()

                AS.specify_phase(cp.iphase_gas)
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.PT_INPUTS, P1, T1)
                if PROFILE_ENABLED:
                    profile["state1_pt"] += perf_counter() - t0
                    t0 = perf_counter()
                h1 = AS.hmass()
                s1 = AS.smass()
                ro1 = AS.rhomass()
                if PROFILE_ENABLED:
                    profile["property_reads"] += perf_counter() - t0

                P1ihx = P1
                T1ihx = T3 - dT_IHX
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.PT_INPUTS, P1ihx, T1ihx)
                if PROFILE_ENABLED:
                    profile["state1ihx_pt"] += perf_counter() - t0
                    t0 = perf_counter()
                h1ihx = AS.hmass()
                s1ihx = AS.smass()
                ro1ihx = AS.rhomass()
                if PROFILE_ENABLED:
                    profile["property_reads"] += perf_counter() - t0

                AS.unspecify_phase()
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.QT_INPUTS, 0, T_cond)
                if PROFILE_ENABLED:
                    profile["state3_qt"] += perf_counter() - t0
                P3 = AS.p()

                AS.specify_phase(cp.iphase_liquid)
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.PT_INPUTS, P3, T3)
                if PROFILE_ENABLED:
                    profile["state3_pt"] += perf_counter() - t0
                    t0 = perf_counter()
                h3 = AS.hmass()
                s3 = AS.smass()
                ro3 = AS.rhomass()
                if PROFILE_ENABLED:
                    profile["property_reads"] += perf_counter() - t0

                AS.unspecify_phase()
                P3ihx = P3
                h3ihx = h3 - (h1ihx - h1)
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.HmassP_INPUTS, h3ihx, P3ihx)
                if PROFILE_ENABLED:
                    profile["state3ihx_hp"] += perf_counter() - t0
                    t0 = perf_counter()
                T3ihx = AS.T()
                s3ihx = AS.smass()
                ro3ihx = AS.rhomass()
                if PROFILE_ENABLED:
                    profile["property_reads"] += perf_counter() - t0

                s2s = s1ihx
                P2s = P3
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.PSmass_INPUTS, P2s, s2s)
                if PROFILE_ENABLED:
                    profile["state2s_ps"] += perf_counter() - t0
                    t0 = perf_counter()
                h2s = AS.hmass()
                T2s = AS.T()
                ro2s = AS.rhomass()
                if PROFILE_ENABLED:
                    profile["property_reads"] += perf_counter() - t0

                h2 = h1ihx + (h2s - h1ihx) / eta_comp
                P2 = P2s
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.HmassP_INPUTS, h2, P2)
                if PROFILE_ENABLED:
                    profile["state2_hp"] += perf_counter() - t0
                    t0 = perf_counter()
                T2 = AS.T()
                s2 = AS.smass()
                ro2 = AS.rhomass()
                if PROFILE_ENABLED:
                    profile["property_reads"] += perf_counter() - t0

                h4 = h3ihx
                P4 = P1
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                AS.update(cp.HmassP_INPUTS, h4, P4)
                if PROFILE_ENABLED:
                    profile["state4_hp"] += perf_counter() - t0
                    t0 = perf_counter()
                T4 = AS.T()
                s4 = AS.smass()
                ro4 = AS.rhomass()
                if PROFILE_ENABLED:
                    profile["property_reads"] += perf_counter() - t0
            except Exception as exc:
                AS.unspecify_phase()
                errors.append(
                    (
                        i,
                        j,
                        k,
                        podil1,
                        podil2,
                        podil3,
                        podil4,
                        str(exc),
                    )
                )
                if PROFILE_ENABLED:
                    profile["worker_errors"] += 1
                continue

            if PROFILE_ENABLED:
                t0 = perf_counter()
            q_in = h1 - h4
            q_out = h2 - h3
            m_dot = Q_out / q_out
            w_cycle = q_out - q_in
            W_comp = w_cycle * m_dot
            COP = q_out / w_cycle
            VHC = q_out * ro1ihx
            if PROFILE_ENABLED:
                profile["cycle_calc"] += perf_counter() - t0

            if PROFILE_ENABLED:
                t0 = perf_counter()
            jk_indices[valid_idx] = (j, k)
            podil_data[valid_idx] = (podil1, podil2, podil3, podil4)
            stav_data[valid_idx] = (
                (P1, T1, h1, s1, ro1),
                (P1ihx, T1ihx, h1ihx, s1ihx, ro1ihx),
                (P2, T2, h2, s2, ro2),
                (P2s, T2s, h2s, s2s, ro2s),
                (P3, T3, h3, s3, ro3),
                (P3ihx, T3ihx, h3ihx, s3ihx, ro3ihx),
                (P4, T4, h4, s4, ro4),
            )
            obeh_data[valid_idx] = (q_in, q_out, m_dot, w_cycle, W_comp, COP, VHC)
            if PROFILE_ENABLED:
                profile["result_store"] += perf_counter() - t0
                profile["worker_points_valid"] += 1
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

    if PROFILE_ENABLED:
        profile["worker_total"] += perf_counter() - worker_start

    return {
        "nazev": f"{latka1}&{latka2}&{latka3}&{latka4}",
        "smes": np.array([latka1, latka2, latka3, latka4], dtype="S"),
        "slices": slices,
        "errors": errors,
        "profile": profile,
    }


def writer(queue, filename, stats_queue):
    profile = empty_profile_stats() if PROFILE_ENABLED else None
    writer_start = perf_counter() if PROFILE_ENABLED else None

    with h5py.File(filename, "w") as f:
        while True:
            item = queue.get()
            if item is None:
                break

            kind = item["kind"]

            if kind == "start":
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                grp = f.create_group(item["nazev"])
                grp.create_dataset("Smes", data=item["smes"])
                grp.create_group("slices")
                grp.create_group("errors")
                if PROFILE_ENABLED:
                    profile["writer_start_group"] += perf_counter() - t0
                    profile["writer_items_start"] += 1
                continue

            if kind == "slice":
                if PROFILE_ENABLED:
                    t0 = perf_counter()
                grp = f[item["nazev"]]["slices"].create_group(f"i_{item['i']:02d}")
                grp.create_dataset("jk_index", data=item["jk"])
                grp.create_dataset("Podil", data=item["podil"], compression="lzf")
                grp.create_dataset("Stav", data=item["stav"], compression="lzf")
                grp.create_dataset("Obeh", data=item["obeh"], compression="lzf")
                if PROFILE_ENABLED:
                    profile["writer_slice_write"] += perf_counter() - t0
                    profile["writer_items_slice"] += 1
                continue

            if kind == "errors":
                if PROFILE_ENABLED:
                    t0 = perf_counter()
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
                if PROFILE_ENABLED:
                    profile["writer_errors_write"] += perf_counter() - t0
                    profile["writer_items_errors"] += 1

    if PROFILE_ENABLED:
        profile["writer_total"] += perf_counter() - writer_start
    stats_queue.put(profile)


with open(os.path.join(cesta1, "směsi", "pure_fluids.txt"), "r") as f:
    latky = [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    profile_main = empty_profile_stats() if PROFILE_ENABLED else None
    main_start = perf_counter() if PROFILE_ENABLED else None
    queue = Queue(maxsize=20)
    stats_queue = Queue(maxsize=1)
    writer_process = Process(target=writer, args=(queue, os.path.join(cesta, "IHX_quat.h5"), stats_queue))
    writer_process.start()
    pool = Pool(cpu_count() - 1)

    mixture_args = list(combinations(latky, 4))
    if PROFILE_ENABLED:
        profile_main["main_mixtures"] = len(mixture_args)

    results_iter = pool.imap_unordered(worker_mixture, mixture_args)
    for _ in range(len(mixture_args)):
        if PROFILE_ENABLED:
            t0 = perf_counter()
        result = next(results_iter)
        if PROFILE_ENABLED:
            profile_main["main_wait_results"] += perf_counter() - t0

        if PROFILE_ENABLED and result["profile"] is not None:
            merge_profile_stats(profile_main, result["profile"])

        if PROFILE_ENABLED:
            t0 = perf_counter()
        queue.put({"kind": "start", "nazev": result["nazev"], "smes": result["smes"]})
        if PROFILE_ENABLED:
            profile_main["main_queue_put"] += perf_counter() - t0

        for slice_result in result["slices"]:
            if PROFILE_ENABLED:
                t0 = perf_counter()
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
            if PROFILE_ENABLED:
                profile_main["main_queue_put"] += perf_counter() - t0

        if result["errors"]:
            if PROFILE_ENABLED:
                t0 = perf_counter()
            error_payload = result["errors"]
            if PROFILE_ENABLED:
                profile_main["main_collect_errors"] += perf_counter() - t0
        else:
            error_payload = []

        if PROFILE_ENABLED:
            t0 = perf_counter()
        queue.put({"kind": "errors", "nazev": result["nazev"], "errors": error_payload})
        if PROFILE_ENABLED:
            profile_main["main_queue_put"] += perf_counter() - t0

    pool.close()
    pool.join()
    queue.put(None)
    writer_profile = stats_queue.get()
    writer_process.join()

    if PROFILE_ENABLED:
        profile_main["main_total"] += perf_counter() - main_start
        merge_profile_stats(profile_main, writer_profile)
        with open(PROFILE_TXT, "w", encoding="utf-8") as f:
            f.write(format_profile_report(profile_main))
