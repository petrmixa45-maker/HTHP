import csv
import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


# ==================================================
# MASTER CONFIGURATION
# ==================================================

NUM_RUNS = 1
MASTER_SEED = 123456789
RESUME_FROM_LAST_RUN = 1
OVERWRITE_EXISTING_RUNS = 0
RESTART_INCOMPLETE_RUNS = 1

RUN_GA_BATCH = 0
RUN_MERGER = 0
RUN_VISUALIZATION = 1

MERGE_ARCHIVES = RUN_MERGER
REMOVE_DUPLICATES = RUN_MERGER
BUILD_MASTER_PARETO = RUN_MERGER
CREATE_PLOTS = RUN_VISUALIZATION
CREATE_STATISTICS = RUN_VISUALIZATION
ANALYZE_RUN_CONTRIBUTIONS = RUN_VISUALIZATION
ANALYZE_COMPONENT_FREQUENCY = RUN_VISUALIZATION
ANALYZE_WEIGHT_DISTRIBUTION = RUN_VISUALIZATION
ANALYZE_CONVERGENCE = RUN_VISUALIZATION
RUN_SELF_TESTS = RUN_VISUALIZATION

STOP_ON_RUN_STAGNATION = 0
STAGNATION_RUN_WINDOW = 10
CHECK_RUN_STAGNATION_DURING_BATCH = STOP_ON_RUN_STAGNATION
USE_STRATIFIED_INITIALIZATION = 0
RUN_TIMEOUT_SECONDS = 0

PLOT_ALL_RUN_SOLUTIONS = 1
PLOT_MASTER_PARETO = 1
PLOT_RUNS_SEPARATELY = 0
WEIGHT_PLOT_XTICK_STEP_PERCENT = 10


# ==================================================
# GA CONFIGURATION PASSED TO BASIC_GA.py
# ==================================================

GA_SCRIPT = "BASIC_GA.py"
GA_PYTHON = sys.executable
FLUIDS_FILE = str(Path("směsi") / "pure_fluids.txt")

COMPONENTS = 5
FRACTION_STEP = 0.02
POPULATION = 100
GENERATIONS = 80
ELITE_SIZE = 4
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.35
FRACTION_MUTATION_STEPS = 1
MEDIUM_MUTATION_RATE = 0.25
LARGE_MUTATION_RATE = 0.05
FLUID_MUTATION_RATE = 0.25
REDISTRIBUTE_MUTATION_RATE = 0.05
RANDOM_IMMIGRANT_RATE = 0.03
EARLY_STOPPING = 20
MAX_ARCHIVE_SIZE = 0
USE_CROSSOVER = 1
USE_GLOBAL_ARCHIVE = 1
INITIAL_FLUID_SET_RATE = 0.25
PROCESSES = max(1, (os.cpu_count() or 2) - 1)

BACKEND = "REFPROP"
T_EVAP = 273.15 + 80.0
T_COND = 273.15 + 130.0
DT_SH = 5.0
DT_SC = 5.0
ETA_COMP = 0.75
Q_OUT = 500000.0


# ==================================================
# OUTPUT STRUCTURE
# ==================================================

# Vysledky se oddeluji podle spousteneho GA scriptu. Pri
# GA_SCRIPT = "BASIC_GA.py" se uklada do results/BASIC_GA, pri
# GA_SCRIPT = "IHX_GA.py" do results/IHX_GA. Diky tomu si neprepisujes
# vysledky ruznych variant tepelneho cerpadla.
RESULTS_ROOT_DIR = Path("results")
GA_RESULTS_NAME = Path(GA_SCRIPT).stem
RESULTS_DIR = RESULTS_ROOT_DIR / GA_RESULTS_NAME
RUNS_DIR = RESULTS_DIR / "runs"
MERGED_DIR = RESULTS_DIR / "merged"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
STATISTICS_DIR = ANALYSIS_DIR / "statistics"
REPORTS_DIR = ANALYSIS_DIR / "reports"

MERGED_ARCHIVES_FILE = MERGED_DIR / "merged_archives.csv"
UNIQUE_ARCHIVES_FILE = MERGED_DIR / "unique_archives.csv"
MASTER_PARETO_FILE = MERGED_DIR / "master_pareto_archive.csv"
DEDUPLICATE_DETAILS_FILE = MERGED_DIR / "duplicate_solution_details.json"
SEED_PLAN_FILE = RESULTS_DIR / "seed_plan.json"


SWITCH_NAMES = [
    "RUN_GA_BATCH",
    "MERGE_ARCHIVES",
    "REMOVE_DUPLICATES",
    "BUILD_MASTER_PARETO",
    "CREATE_PLOTS",
    "CREATE_STATISTICS",
    "ANALYZE_RUN_CONTRIBUTIONS",
    "ANALYZE_COMPONENT_FREQUENCY",
    "ANALYZE_WEIGHT_DISTRIBUTION",
    "ANALYZE_CONVERGENCE",
    "RUN_SELF_TESTS",
    "STOP_ON_RUN_STAGNATION",
    "CHECK_RUN_STAGNATION_DURING_BATCH",
    "RESUME_FROM_LAST_RUN",
    "OVERWRITE_EXISTING_RUNS",
    "RESTART_INCOMPLETE_RUNS",
    "USE_STRATIFIED_INITIALIZATION",
    "PLOT_ALL_RUN_SOLUTIONS",
    "PLOT_MASTER_PARETO",
    "PLOT_RUNS_SEPARATELY",
    "USE_CROSSOVER",
    "USE_GLOBAL_ARCHIVE",
]


def ensure_directories() -> None:
    for directory in [
        RUNS_DIR,
        MERGED_DIR,
        PLOTS_DIR,
        STATISTICS_DIR,
        REPORTS_DIR,
        PLOTS_DIR / "weights",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def validate_switches() -> None:
    for name in SWITCH_NAMES:
        value = globals()[name]
        if value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1, got {value!r}.")
    if NUM_RUNS < 1:
        raise ValueError(f"NUM_RUNS must be >= 1, got {NUM_RUNS!r}.")
    if RUN_TIMEOUT_SECONDS < 0:
        raise ValueError(
            f"RUN_TIMEOUT_SECONDS must be >= 0, got {RUN_TIMEOUT_SECONDS!r}."
        )
    if RUN_GA_BATCH:
        if not (SCRIPT_DIR / GA_SCRIPT).exists():
            raise FileNotFoundError(f"GA_SCRIPT does not exist: {GA_SCRIPT}")
        if not (SCRIPT_DIR / FLUIDS_FILE).exists():
            raise FileNotFoundError(f"FLUIDS_FILE does not exist: {FLUIDS_FILE}")


def run_dir(run_id: int) -> Path:
    return RUNS_DIR / f"run_{run_id:04d}"


def parse_run_id_from_dir(path: Path) -> Optional[int]:
    """Return numeric run id from a path named run_0001, otherwise None."""
    if not path.name.startswith("run_"):
        return None
    suffix = path.name[4:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def existing_run_ids() -> List[int]:
    """Find already written run directories in results/runs."""
    if not RUNS_DIR.exists():
        return []

    run_ids = []
    for path in RUNS_DIR.iterdir():
        if not path.is_dir():
            continue
        run_id = parse_run_id_from_dir(path)
        if run_id is not None:
            run_ids.append(run_id)
    return sorted(run_ids)


def is_completed_run(run_id: int) -> bool:
    """Return True only for a run that finished and wrote usable output files."""
    directory = run_dir(run_id)
    metadata_file = directory / "metadata.json"
    if not metadata_file.exists():
        return False

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    status = metadata.get("status")
    return_code = int(metadata.get("return_code", -1))
    if return_code != 0:
        return False
    if status is not None and status != "completed":
        return False

    pareto_file = resolve_run_output_file(
        metadata, metadata_file, "pareto_file", "pareto_archive.csv"
    )
    history_file = resolve_run_output_file(
        metadata, metadata_file, "history_file", "ga_history.csv"
    )

    return pareto_file.exists() and history_file.exists()


def completed_run_ids() -> List[int]:
    """Find run ids that are truly completed, not just present as directories."""
    return [run_id for run_id in existing_run_ids() if is_completed_run(run_id)]


def next_run_id() -> int:
    """Next run id for resume mode, based on completed runs only."""
    ids = completed_run_ids()
    return ids[-1] + 1 if ids else 1


def run_can_be_skipped(run_id: int) -> bool:
    """Decide whether an existing run directory should be skipped or retried."""
    if not run_dir(run_id).exists():
        return False
    if OVERWRITE_EXISTING_RUNS:
        return False
    if is_completed_run(run_id):
        return True
    return not RESTART_INCOMPLETE_RUNS


def derived_seed_info(master_seed: int, run_id: int) -> Dict[str, object]:
    """Create a deterministic independent seed using NumPy SeedSequence.

    This avoids the weaker BASE_SEED + run_id pattern. The run_id is encoded in
    the SeedSequence spawn_key, so a specific run can be reproduced directly
    without generating all previous runs first.
    """
    seed_sequence = np.random.SeedSequence(master_seed, spawn_key=(run_id - 1,))
    derived_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
    return {
        "run_id": run_id,
        "master_seed": int(master_seed),
        "derived_seed": derived_seed,
        "spawn_key": list(seed_sequence.spawn_key),
        "entropy": int(seed_sequence.entropy),
    }


def create_seed_plan(start_run_id: int, num_runs: int) -> List[Dict[str, object]]:
    """Create seed plan for this batch.

    NUM_RUNS means how many new runs to execute now. If start_run_id is 6 and
    num_runs is 5, the batch will run ids 6, 7, 8, 9, 10.
    """
    end_run_id = start_run_id + num_runs - 1
    plan = [
        derived_seed_info(MASTER_SEED, run_id)
        for run_id in range(start_run_id, end_run_id + 1)
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEED_PLAN_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "master_seed": MASTER_SEED,
                "num_new_runs": num_runs,
                "start_run_id": start_run_id,
                "end_run_id": end_run_id,
                "runs": plan,
            },
            f,
            indent=2,
        )
    batch_seed_plan_file = RESULTS_DIR / f"seed_plan_runs_{start_run_id:04d}_{end_run_id:04d}.json"
    with open(batch_seed_plan_file, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    return plan


def read_fluids(filename: str) -> List[str]:
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stratified_component_set(run_id: int, fluids: Sequence[str]) -> Optional[Tuple[str, ...]]:
    if not USE_STRATIFIED_INITIALIZATION:
        return None
    if COMPONENTS > len(fluids):
        raise ValueError("COMPONENTS is larger than number of fluids.")

    component_sets = list(combinations(sorted(fluids), COMPONENTS))
    rng = np.random.default_rng(
        np.random.SeedSequence(MASTER_SEED, spawn_key=(999999,))
    )
    order = rng.permutation(len(component_sets))
    return tuple(component_sets[int(order[(run_id - 1) % len(order)])])


def ga_command(
    run_id: int,
    seed: int,
    pareto_file: Path,
    history_file: Path,
    initial_fluid_set: Optional[Sequence[str]],
) -> List[str]:
    command = [
        GA_PYTHON,
        GA_SCRIPT,
        "--fluids-file",
        FLUIDS_FILE,
        "--components",
        str(COMPONENTS),
        "--fraction-step",
        str(FRACTION_STEP),
        "--population",
        str(POPULATION),
        "--generations",
        str(GENERATIONS),
        "--elite-size",
        str(ELITE_SIZE),
        "--tournament-size",
        str(TOURNAMENT_SIZE),
        "--mutation-rate",
        str(MUTATION_RATE),
        "--fraction-mutation-steps",
        str(FRACTION_MUTATION_STEPS),
        "--medium-mutation-rate",
        str(MEDIUM_MUTATION_RATE),
        "--large-mutation-rate",
        str(LARGE_MUTATION_RATE),
        "--fluid-mutation-rate",
        str(FLUID_MUTATION_RATE),
        "--redistribute-mutation-rate",
        str(REDISTRIBUTE_MUTATION_RATE),
        "--random-immigrant-rate",
        str(RANDOM_IMMIGRANT_RATE),
        "--early-stopping",
        str(EARLY_STOPPING),
        "--processes",
        str(PROCESSES),
        "--backend",
        BACKEND,
        "--T-evap",
        str(T_EVAP),
        "--T-cond",
        str(T_COND),
        "--dT-SH",
        str(DT_SH),
        "--dT-SC",
        str(DT_SC),
        "--eta-comp",
        str(ETA_COMP),
        "--Q-out",
        str(Q_OUT),
        "--seed",
        str(seed),
        "--history-file",
        str(history_file),
        "--pareto-file",
        str(pareto_file),
    ]

    if MAX_ARCHIVE_SIZE > 0:
        command.extend(["--max-archive-size", str(MAX_ARCHIVE_SIZE)])
    if not USE_CROSSOVER:
        command.append("--no-crossover")
    if not USE_GLOBAL_ARCHIVE:
        command.append("--no-global-archive")
    if initial_fluid_set:
        command.extend(["--initial-fluid-set", ",".join(initial_fluid_set)])
        command.extend(["--initial-fluid-set-rate", str(INITIAL_FLUID_SET_RATE)])

    return command


def read_csv_rows(filename: Path) -> List[Dict[str, str]]:
    if not filename.exists():
        return []
    with open(filename, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(filename: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def resolve_run_output_file(
    metadata: Dict[str, object],
    metadata_file: Path,
    field: str,
    default_name: str,
) -> Path:
    """Resolve output paths robustly, including old metadata moved by hand.

    Older runs can contain paths like results/runs/run_0100/pareto_archive.csv
    even after their whole run directory was moved to results/BASIC_GA/runs.
    Prefer the stored path when it exists, otherwise fall back to the file next
    to metadata.json.
    """
    stored = metadata.get(field)
    candidates: List[Path] = []

    if stored:
        stored_path = Path(str(stored))
        candidates.append(stored_path if stored_path.is_absolute() else SCRIPT_DIR / stored_path)

    candidates.append(metadata_file.parent / default_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def last_history_row(history_file: Path) -> Dict[str, str]:
    rows = read_csv_rows(history_file)
    return rows[-1] if rows else {}


def summarize_history(history_file: Path) -> Dict[str, int]:
    """Read GA history and return whole-run evaluation/cache totals."""
    rows = read_csv_rows(history_file)
    if not rows:
        return {
            "num_unique_evaluations": 0,
            "num_cache_hits": 0,
            "num_cache_misses": 0,
            "num_duplicate_requests": 0,
            "num_fitness_requests": 0,
        }

    final = rows[-1]
    return {
        "num_unique_evaluations": int(
            float(final.get("total_actual_evaluations", final.get("unique_mixtures", 0)) or 0)
        ),
        "num_cache_hits": int(sum(float(row.get("cache_hits", 0) or 0) for row in rows)),
        "num_cache_misses": int(sum(float(row.get("cache_misses", 0) or 0) for row in rows)),
        "num_duplicate_requests": int(
            sum(float(row.get("duplicate_requests", 0) or 0) for row in rows)
        ),
        "num_fitness_requests": int(
            sum(float(row.get("fitness_requests", 0) or 0) for row in rows)
        ),
    }


def run_command_streamed(command: Sequence[str], log_file: Path) -> int:
    """Run one GA process and stream its output directly to log_file.

    This avoids keeping the whole stdout/stderr of a long REFPROP run in RAM.
    """
    timeout = None if RUN_TIMEOUT_SECONDS <= 0 else RUN_TIMEOUT_SECONDS
    with open(log_file, "w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                list(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=SCRIPT_DIR,
            )
        except OSError as exc:
            log.write(f"FAILED TO START GA PROCESS: {exc}\n")
            return 127
        try:
            return process.wait(timeout=timeout)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.write("\nINTERRUPTED: parent process received KeyboardInterrupt.\n")
            raise
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.write(
                f"\nTIMEOUT: run exceeded RUN_TIMEOUT_SECONDS={RUN_TIMEOUT_SECONDS}.\n"
            )
            return 124


def maybe_update_batch_stagnation(
    current_master_rows: Sequence[Dict[str, object]],
    archive_rows: Sequence[Dict[str, str]],
    seed_record: Dict[str, object],
) -> Tuple[List[Dict[str, object]], int]:
    """Update an in-memory master front for optional during-batch stagnation checks."""
    tagged_archive_rows = [
        {
            **row,
            "run_id": seed_record["run_id"],
            "master_seed": seed_record["master_seed"],
            "derived_seed": seed_record["derived_seed"],
        }
        for row in archive_rows
    ]
    previous_keys = {canonical_solution_key(row) for row in current_master_rows}
    combined_rows = list(current_master_rows) + tagged_archive_rows
    unique_rows = remove_duplicate_solutions(combined_rows, output_file=None)
    new_master_rows = pareto_filter_rows(unique_rows)
    new_keys = {canonical_solution_key(row) for row in new_master_rows}
    return new_master_rows, len(new_keys - previous_keys)


def run_ga_batch() -> None:
    fluids = read_fluids(FLUIDS_FILE)
    start_run_id = next_run_id() if RESUME_FROM_LAST_RUN else 1
    end_run_id = start_run_id + NUM_RUNS - 1
    seed_plan = create_seed_plan(start_run_id, NUM_RUNS)
    batch_master_rows: List[Dict[str, object]] = []
    recent_master_contributions: List[int] = []

    logging.info(
        "Starting GA batch: NUM_RUNS=%d, start_run_id=%04d, end_run_id=%04d",
        NUM_RUNS,
        start_run_id,
        end_run_id,
    )

    for seed_record in seed_plan:
        run_id = int(seed_record["run_id"])
        directory = run_dir(run_id)
        if run_can_be_skipped(run_id):
            logging.warning(
                "Skipping completed existing %s because OVERWRITE_EXISTING_RUNS=0.",
                directory,
            )
            continue
        directory.mkdir(parents=True, exist_ok=True)

        pareto_file = directory / "pareto_archive.csv"
        history_file = directory / "ga_history.csv"
        metadata_file = directory / "metadata.json"
        log_file = directory / "log.txt"

        metadata = {
            **seed_record,
            "initial_fluid_set": None,
            "status": "preparing",
            "start_time": datetime.now().isoformat(timespec="seconds"),
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        try:
            initial_set = stratified_component_set(run_id, fluids)
            metadata.update(
                {
                    "initial_fluid_set": list(initial_set) if initial_set else None,
                    "status": "running",
                    "command": ga_command(
                        run_id,
                        int(seed_record["derived_seed"]),
                        pareto_file,
                        history_file,
                        initial_set,
                    ),
                }
            )
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            start = time.perf_counter()
            return_code = run_command_streamed(metadata["command"], log_file)
            runtime = time.perf_counter() - start

            history_summary = summarize_history(history_file)
            archive_rows = read_csv_rows(pareto_file)

            metadata.update(
                {
                    "end_time": datetime.now().isoformat(timespec="seconds"),
                    "runtime_seconds": runtime,
                    "return_code": return_code,
                    "status": (
                        "completed"
                        if return_code == 0
                        else "timeout"
                        if return_code == 124
                        else "failed"
                    ),
                    **history_summary,
                    "final_archive_size": len(archive_rows),
                    "history_file": str(history_file),
                    "pareto_file": str(pareto_file),
                    "log_file": str(log_file),
                }
            )
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as exc:
            runtime = 0.0
            return_code = 1
            archive_rows = []
            history_summary = summarize_history(history_file)
            metadata.update(
                {
                    "end_time": datetime.now().isoformat(timespec="seconds"),
                    "runtime_seconds": runtime,
                    "return_code": return_code,
                    "status": "failed_orchestrator",
                    "error": repr(exc),
                    **history_summary,
                    "final_archive_size": 0,
                    "history_file": str(history_file),
                    "pareto_file": str(pareto_file),
                    "log_file": str(log_file),
                }
            )
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\nORCHESTRATOR ERROR: {exc!r}\n")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        logging.info(
            "Run %04d finished: return_code=%d, archive=%d, unique_evaluations=%d",
            run_id,
            return_code,
            len(archive_rows),
            metadata["num_unique_evaluations"],
        )

        if return_code != 0:
            logging.warning("Run %04d failed. See %s", run_id, log_file)
            del archive_rows, history_summary, metadata
            gc.collect()
            continue

        should_stop_batch = False
        if CHECK_RUN_STAGNATION_DURING_BATCH:
            batch_master_rows, new_master_count = maybe_update_batch_stagnation(
                batch_master_rows,
                archive_rows,
                seed_record,
            )
            recent_master_contributions.append(new_master_count)

            if len(recent_master_contributions) >= STAGNATION_RUN_WINDOW:
                recent = recent_master_contributions[-STAGNATION_RUN_WINDOW:]
                if all(count == 0 for count in recent):
                    logging.warning(
                        "Master Pareto archive has not improved during the last %d runs.",
                        STAGNATION_RUN_WINDOW,
                    )
                    if STOP_ON_RUN_STAGNATION:
                        should_stop_batch = True

        del archive_rows, history_summary, metadata
        gc.collect()

        if should_stop_batch:
            break


def parse_solution(row: Dict[str, str]) -> Tuple[List[str], List[float]]:
    fluids = [item.strip() for item in row["fluids"].split("&") if item.strip()]
    fractions = [float(item) for item in row["fractions"].split(";") if item.strip()]
    if len(fluids) != len(fractions):
        raise ValueError(f"Invalid solution row: {row}")
    return fluids, fractions


def canonical_solution_key(row: Dict[str, str], fraction_step: float = FRACTION_STEP) -> Tuple[Tuple[str, int], ...]:
    fluids, fractions = parse_solution(row)
    units = [int(round(fraction / fraction_step)) for fraction in fractions]
    return tuple(sorted(zip(fluids, units), key=lambda item: item[0]))


def row_objectives(row: Dict[str, str]) -> Tuple[float, float, float]:
    return float(row["COP"]), float(row["VHC"]), float(row["W_comp"])


def dominates_row(row_a: Dict[str, str], row_b: Dict[str, str]) -> bool:
    cop_a, vhc_a, w_a = row_objectives(row_a)
    cop_b, vhc_b, w_b = row_objectives(row_b)
    return (
        cop_a >= cop_b
        and vhc_a >= vhc_b
        and w_a <= w_b
        and (cop_a > cop_b or vhc_a > vhc_b or w_a < w_b)
    )


def merge_archives() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metadata_file in sorted(RUNS_DIR.glob("run_*/metadata.json")):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        archive_file = resolve_run_output_file(
            metadata, metadata_file, "pareto_file", "pareto_archive.csv"
        )
        for row in read_csv_rows(archive_file):
            rows.append(
                {
                    **row,
                    "run_id": metadata["run_id"],
                    "master_seed": metadata["master_seed"],
                    "derived_seed": metadata["derived_seed"],
                }
            )

    fieldnames = [
        "run_id",
        "master_seed",
        "derived_seed",
        "fluids",
        "fractions",
        "COP",
        "VHC",
        "W_comp",
        "pareto_rank",
        "crowding_distance",
    ]
    write_csv_rows(MERGED_ARCHIVES_FILE, rows, fieldnames)
    logging.info("Merged %d rows into %s", len(rows), MERGED_ARCHIVES_FILE)
    return rows


def remove_duplicate_solutions(
    rows: Optional[Sequence[Dict[str, object]]] = None,
    output_file: Optional[Path] = UNIQUE_ARCHIVES_FILE,
) -> List[Dict[str, object]]:
    if rows is None:
        rows = read_csv_rows(MERGED_ARCHIVES_FILE)

    grouped: Dict[Tuple[Tuple[str, int], ...], Dict[str, object]] = {}
    found_runs: Dict[Tuple[Tuple[str, int], ...], set] = {}
    found_seeds: Dict[Tuple[Tuple[str, int], ...], set] = {}

    for raw_row in rows:
        row = dict(raw_row)
        key = canonical_solution_key(row)
        found_runs.setdefault(key, set()).add(str(row.get("run_id", "")))
        found_seeds.setdefault(key, set()).add(str(row.get("derived_seed", "")))

        if key not in grouped:
            grouped[key] = row
        elif dominates_row(row, grouped[key]):
            grouped[key] = row

    unique_rows = []
    for key, row in grouped.items():
        row = dict(row)
        row["canonical_key"] = json.dumps(list(key))
        row["found_in_runs"] = ";".join(sorted(found_runs[key], key=lambda value: int(value or 0)))
        row["found_in_seeds"] = ";".join(sorted(found_seeds[key]))
        row["duplicate_count"] = len(found_runs[key])
        unique_rows.append(row)

    fieldnames = [
        "canonical_key",
        "found_in_runs",
        "found_in_seeds",
        "duplicate_count",
        "run_id",
        "master_seed",
        "derived_seed",
        "fluids",
        "fractions",
        "COP",
        "VHC",
        "W_comp",
        "pareto_rank",
        "crowding_distance",
    ]
    if output_file is not None:
        write_csv_rows(output_file, unique_rows, fieldnames)
        details = {
            json.dumps(list(key)): {
                "found_in_runs": sorted(found_runs[key], key=lambda value: int(value or 0)),
                "found_in_seeds": sorted(found_seeds[key]),
                "duplicate_count": len(found_runs[key]),
            }
            for key in grouped
        }
        with open(DEDUPLICATE_DETAILS_FILE, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)
        logging.info("Wrote %d unique rows to %s", len(unique_rows), output_file)
    return unique_rows


def pareto_filter_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    nondominated = []
    for i, row_i in enumerate(rows):
        dominated = False
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            if dominates_row(row_j, row_i):
                dominated = True
                break
        if not dominated:
            nondominated.append(dict(row_i))
    return nondominated


def build_master_pareto_archive(rows: Optional[Sequence[Dict[str, object]]] = None) -> List[Dict[str, object]]:
    if rows is None:
        rows = read_csv_rows(UNIQUE_ARCHIVES_FILE)
    master_rows = pareto_filter_rows(rows)
    for row in master_rows:
        row["archive_type"] = "empirical_master_pareto_front"

    fieldnames = [
        "archive_type",
        "canonical_key",
        "found_in_runs",
        "found_in_seeds",
        "duplicate_count",
        "run_id",
        "master_seed",
        "derived_seed",
        "fluids",
        "fractions",
        "COP",
        "VHC",
        "W_comp",
        "pareto_rank",
        "crowding_distance",
    ]
    write_csv_rows(MASTER_PARETO_FILE, master_rows, fieldnames)
    logging.info("Wrote %d master Pareto rows to %s", len(master_rows), MASTER_PARETO_FILE)
    return master_rows


def analyze_run_contributions() -> None:
    merged_rows = read_csv_rows(MERGED_ARCHIVES_FILE)
    rows_by_run: Dict[int, List[Dict[str, str]]] = {}
    for row in merged_rows:
        rows_by_run.setdefault(int(row["run_id"]), []).append(row)

    seen_unique = set()
    cumulative_rows: List[Dict[str, object]] = []
    previous_master_keys = set()
    report_rows = []

    for run_id in sorted(rows_by_run):
        run_rows = rows_by_run[run_id]
        run_keys = {canonical_solution_key(row) for row in run_rows}
        new_unique = run_keys - seen_unique
        seen_unique.update(run_keys)

        cumulative_rows.extend(run_rows)
        cumulative_unique = remove_duplicate_solutions(cumulative_rows, output_file=None)
        cumulative_master = pareto_filter_rows(cumulative_unique)
        master_keys = {canonical_solution_key(row) for row in cumulative_master}
        new_master = master_keys - previous_master_keys
        previous_master_keys = master_keys

        report_rows.append(
            {
                "run_id": run_id,
                "new_unique_solutions": len(new_unique),
                "new_master_pareto_solutions": len(new_master),
                "cumulative_unique_solutions": len(seen_unique),
                "cumulative_master_pareto_size": len(master_keys),
            }
        )

    output_file = STATISTICS_DIR / "run_contributions.csv"
    write_csv_rows(
        output_file,
        report_rows,
        [
            "run_id",
            "new_unique_solutions",
            "new_master_pareto_solutions",
            "cumulative_unique_solutions",
            "cumulative_master_pareto_size",
        ],
    )

    if len(report_rows) >= STAGNATION_RUN_WINDOW:
        recent = report_rows[-STAGNATION_RUN_WINDOW:]
        if all(int(row["new_master_pareto_solutions"]) == 0 for row in recent):
            logging.warning(
                "Master Pareto archive has not improved during the last %d runs.",
                STAGNATION_RUN_WINDOW,
            )


def create_statistics() -> None:
    master_rows = read_csv_rows(MASTER_PARETO_FILE)
    master_keys = {canonical_solution_key(row) for row in master_rows}
    stats_rows = []

    for metadata_file in sorted(RUNS_DIR.glob("run_*/metadata.json")):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        archive_file = resolve_run_output_file(
            metadata, metadata_file, "pareto_file", "pareto_archive.csv"
        )
        archive_rows = read_csv_rows(archive_file)
        if archive_rows:
            cops = [float(row["COP"]) for row in archive_rows]
            vhcs = [float(row["VHC"]) for row in archive_rows]
            works = [float(row["W_comp"]) for row in archive_rows]
        else:
            cops, vhcs, works = [np.nan], [np.nan], [np.nan]

        run_keys = {canonical_solution_key(row) for row in archive_rows}
        stats_rows.append(
            {
                "run_id": metadata["run_id"],
                "seed": metadata["derived_seed"],
                "archive_size": len(archive_rows),
                "unique_solutions": len(run_keys),
                "solutions_surviving_in_master": len(run_keys & master_keys),
                "best_COP": np.nanmax(cops),
                "best_VHC": np.nanmax(vhcs),
                "best_W_comp": np.nanmin(works),
                "runtime": metadata.get("runtime_seconds", ""),
                "unique_evaluations": metadata.get("num_unique_evaluations", ""),
            }
        )

    write_csv_rows(
        STATISTICS_DIR / "run_statistics.csv",
        stats_rows,
        [
            "run_id",
            "seed",
            "archive_size",
            "unique_solutions",
            "solutions_surviving_in_master",
            "best_COP",
            "best_VHC",
            "best_W_comp",
            "runtime",
            "unique_evaluations",
        ],
    )


def component_frequency(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        fluids, _ = parse_solution(row)
        for fluid in fluids:
            counts[fluid] = counts.get(fluid, 0) + 1
    return counts


def analyze_component_frequency() -> None:
    all_rows = read_csv_rows(UNIQUE_ARCHIVES_FILE)
    master_rows = read_csv_rows(MASTER_PARETO_FILE)
    rows = []
    all_counts = component_frequency(all_rows)
    master_counts = component_frequency(master_rows)
    fluids = sorted(set(all_counts) | set(master_counts))

    for fluid in fluids:
        rows.append(
            {
                "fluid": fluid,
                "all_archives_count": all_counts.get(fluid, 0),
                "all_archives_percent": 100.0 * all_counts.get(fluid, 0) / max(1, len(all_rows)),
                "master_count": master_counts.get(fluid, 0),
                "master_percent": 100.0 * master_counts.get(fluid, 0) / max(1, len(master_rows)),
            }
        )

    write_csv_rows(
        STATISTICS_DIR / "component_frequency.csv",
        rows,
        ["fluid", "all_archives_count", "all_archives_percent", "master_count", "master_percent"],
    )


def analyze_weight_distribution() -> None:
    rows = read_csv_rows(MASTER_PARETO_FILE)
    by_fluid: Dict[str, List[float]] = {}
    for row in rows:
        fluids, fractions = parse_solution(row)
        for fluid, fraction in zip(fluids, fractions):
            by_fluid.setdefault(fluid, []).append(100.0 * fraction)

    summary_rows = []
    for fluid, weights in sorted(by_fluid.items()):
        summary_rows.append(
            {
                "fluid": fluid,
                "count": len(weights),
                "mean_weight_percent": float(np.mean(weights)),
                "min_weight_percent": float(np.min(weights)),
                "max_weight_percent": float(np.max(weights)),
            }
        )
    write_csv_rows(
        STATISTICS_DIR / "weight_distribution_summary.csv",
        summary_rows,
        ["fluid", "count", "mean_weight_percent", "min_weight_percent", "max_weight_percent"],
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is not installed; skipping weight histograms.")
        return

    for fluid, weights in sorted(by_fluid.items()):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        bins = np.arange(0, 102, FRACTION_STEP * 100.0)
        ax.hist(weights, bins=bins, edgecolor="black")
        ax.set_xlabel("Mixture fraction [%]")
        ax.set_ylabel("Count")
        ax.set_title(f"Weight distribution for {fluid}")
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, WEIGHT_PLOT_XTICK_STEP_PERCENT))
        ax.grid(True, linestyle=":", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "weights" / f"{fluid}_weights.png", dpi=160)
        plt.close(fig)


def cop_vhc_plot(
    rows_all: Sequence[Dict[str, str]],
    rows_master: Sequence[Dict[str, str]],
    filename: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)

    if PLOT_ALL_RUN_SOLUTIONS and rows_all:
        ax.scatter(
            [float(row["VHC"]) / 1_000_000.0 for row in rows_all],
            [float(row["COP"]) for row in rows_all],
            color="tab:blue",
            s=16,
            linewidths=0,
            label="All unique archive solutions",
        )

    if PLOT_MASTER_PARETO and rows_master:
        ax.scatter(
            [float(row["VHC"]) / 1_000_000.0 for row in rows_master],
            [float(row["COP"]) for row in rows_master],
            color="tab:orange",
            s=34,
            edgecolors="black",
            linewidths=0.35,
            label="Empirical master Pareto front",
        )

    ax.set_title(f"COP vs. volumetric heating capacity - {GA_RESULTS_NAME}")
    ax.set_xlabel("Volumetric heating capacity [MJ/m3]")
    ax.set_ylabel("COP [-]")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()

    ax.text(
        0.01,
        0.99,
        (
            f"{len(rows_all):,} unique archive solutions | "
            f"{len(rows_master):,} master Pareto solutions"
        ).replace(",", " "),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8"},
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_plots() -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        logging.warning("matplotlib is not installed; skipping plots.")
        return

    all_rows = read_csv_rows(UNIQUE_ARCHIVES_FILE)
    master_rows = read_csv_rows(MASTER_PARETO_FILE)
    cop_vhc_plot(all_rows, master_rows, PLOTS_DIR / "cop_vs_vhc.png")

    if PLOT_RUNS_SEPARATELY:
        merged_rows = read_csv_rows(MERGED_ARCHIVES_FILE)
        for run_id in sorted({row["run_id"] for row in merged_rows}):
            run_rows = [row for row in merged_rows if row["run_id"] == run_id]
            run_prefix = PLOTS_DIR / f"run_{int(run_id):04d}"
            cop_vhc_plot(run_rows, [], Path(f"{run_prefix}_cop_vs_vhc.png"))


def analyze_convergence() -> None:
    rows = []
    for metadata_file in sorted(RUNS_DIR.glob("run_*/metadata.json")):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        history_file = resolve_run_output_file(
            metadata, metadata_file, "history_file", "ga_history.csv"
        )
        history_rows = read_csv_rows(history_file)
        for row in history_rows:
            rows.append(
                {
                    "run_id": metadata["run_id"],
                    "seed": metadata["derived_seed"],
                    **row,
                }
            )
    if rows:
        write_csv_rows(
            STATISTICS_DIR / "convergence_history.csv",
            rows,
            list(rows[0].keys()),
        )


def self_tests() -> None:
    seed_a = derived_seed_info(123, 7)["derived_seed"]
    seed_b = derived_seed_info(123, 7)["derived_seed"]
    assert seed_a == seed_b
    assert parse_run_id_from_dir(Path("run_0005")) == 5
    assert parse_run_id_from_dir(Path("results") / "runs" / "run_0010") == 10
    assert parse_run_id_from_dir(Path("run_latest")) is None
    run6_direct = derived_seed_info(123456789, 6)["derived_seed"]
    run6_from_plan = [
        derived_seed_info(123456789, run_id)
        for run_id in range(6, 11)
    ][0]["derived_seed"]
    assert run6_direct == run6_from_plan

    row_a = {"fluids": "A&B", "fractions": "0.40;0.60", "COP": "10", "VHC": "35", "W_comp": "100"}
    row_b = {"fluids": "B&A", "fractions": "0.60;0.40", "COP": "10", "VHC": "35", "W_comp": "100"}
    assert canonical_solution_key(row_a) == canonical_solution_key(row_b)

    duplicate_rows = [
        {"run_id": "1", "derived_seed": "11", **row_a},
        {"run_id": "2", "derived_seed": "22", **row_b},
    ]
    unique = remove_duplicate_solutions(duplicate_rows, output_file=None)
    assert len(unique) == 1
    assert unique[0]["found_in_runs"] == "1;2"

    dominated = {"fluids": "A&B", "fractions": "0.38;0.62", "COP": "9", "VHC": "34", "W_comp": "110"}
    assert dominates_row(row_a, dominated)
    assert len(pareto_filter_rows([row_a, dominated])) == 1

    metadata = {
        "run_id": 1,
        "master_seed": 123,
        "derived_seed": seed_a,
        "start_time": "x",
        "end_time": "y",
        "runtime_seconds": 0.1,
    }
    assert metadata["run_id"] == 1 and metadata["derived_seed"] == seed_a
    logging.info("Self tests passed.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    validate_switches()
    ensure_directories()

    if RUN_SELF_TESTS:
        self_tests()

    merged_rows = None
    unique_rows = None

    if RUN_GA_BATCH:
        run_ga_batch()

    if MERGE_ARCHIVES:
        merged_rows = merge_archives()

    if REMOVE_DUPLICATES:
        unique_rows = remove_duplicate_solutions(merged_rows)

    if BUILD_MASTER_PARETO:
        build_master_pareto_archive(unique_rows)

    if ANALYZE_RUN_CONTRIBUTIONS:
        analyze_run_contributions()

    if CREATE_STATISTICS:
        create_statistics()

    if ANALYZE_COMPONENT_FREQUENCY:
        analyze_component_frequency()

    if ANALYZE_WEIGHT_DISTRIBUTION:
        analyze_weight_distribution()

    if ANALYZE_CONVERGENCE:
        analyze_convergence()

    if CREATE_PLOTS:
        create_plots()


if __name__ == "__main__":
    main()
