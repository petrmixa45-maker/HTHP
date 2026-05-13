import argparse
import logging
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


os.environ.setdefault("RPPREFIX", r"C:\Program Files\REFPROP")

PENALTY_VALUE = 1.0e6
FRACTION_STEP = 0.02
komponenty = 5


@dataclass
class CycleParams:
    # Parametry tepelného oběhu. Tyto hodnoty uprav, pokud chceš měnit
    # provozní bod výparníku/kondenzátoru.
    T_evap: float = 273.15 + 80.0
    T_cond: float = 273.15 + 130.0
    dT_SH: float = 5.0
    dT_SC: float = 5.0
    eta_comp: float = 0.75
    Q_out: float = 500000.0
    backend: str = "REFPROP"


@dataclass
class Individual:
    # Jeden jedinec GA = konkrétní sada látek a konkrétní složení.
    # fractions jsou molární podíly, např. [0.20, 0.35, 0.15, 0.30].
    fluids: Tuple[str, ...]
    fractions: np.ndarray
    fitness: Optional[float] = None
    pareto_rank: int = 10**9
    crowding_distance: float = 0.0


@dataclass
class Evaluation:
    # Výsledek jednoho volání CoolProp/REFPROP. Ukládá se do cache,
    # aby se stejná směs nepočítala opakovaně.
    # Cíle Pareto optimalizace:
    #   1) maximalizovat COP
    #   2) maximalizovat VHC
    #   3) minimalizovat W_comp
    key: Tuple
    fitness: float
    COP: float = np.nan
    VHC: float = np.nan
    W_comp: float = np.nan
    error: str = ""


def load_fluids(filename: str) -> List[str]:
    """Načte seznam povolených čistých látek ze souboru pure_fluids.txt."""
    with open(filename, "r", encoding="utf-8") as f:
        fluids = [line.strip() for line in f if line.strip()]

    if not fluids:
        raise ValueError(f"Soubor {filename!r} neobsahuje zadne latky.")
    return fluids


def normalize_fractions(fractions: Sequence[float]) -> np.ndarray:
    """Normalizuje podíly na součet 1 a ohlídá, že žádný podíl není nulový."""
    fractions = np.asarray(fractions, dtype=np.float64)
    if np.any(fractions <= 0.0):
        raise ValueError("Kazda slozka musi mit kladny molarni podil.")
    total = float(np.sum(fractions))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Podily nelze normalizovat.")
    return fractions / total


def fraction_units(fraction_step: float) -> int:
    """Vrátí počet dílků v celé směsi; pro krok 0.02 je to 50 dílků."""
    units = int(round(1.0 / fraction_step))
    if not np.isclose(units * fraction_step, 1.0):
        raise ValueError("fraction_step musi presne delit 1.0, napr. 0.05.")
    return units


def units_to_fractions(units: Sequence[int], fraction_step: float) -> np.ndarray:
    """Převede celočíselné dílky na molární podíly."""
    units = np.asarray(units, dtype=np.int64)
    if np.any(units <= 0):
        raise ValueError("Kazda slozka musi mit alespon jeden dil podilu.")
    if int(np.sum(units)) != fraction_units(fraction_step):
        raise ValueError("Soucet dilku podilu musi odpovidat 100 %.")
    return units.astype(np.float64) * fraction_step


def quantize_fractions(
    fractions: Sequence[float], n_components: int, fraction_step: float
) -> np.ndarray:
    """Zaokrouhlí složení na kroky po fraction_step a zachová součet 100 %.

    GA může při crossoveru vytvořit mezihodnoty, např. 0.173. Tady se převedou
    na nejbližší použitelné složení tak, aby každá složka měla alespoň jeden
    dílek, tedy při kroku 2 % minimálně 2 %.
    """
    total_units = fraction_units(fraction_step)
    if n_components > total_units:
        raise ValueError(
            "Pocet slozek je prilis vysoky pro zvoleny krok podilu. "
            f"Pri kroku {fraction_step:.3f} lze mit maximalne {total_units} slozek."
        )

    weights = np.asarray(fractions, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(n_components, dtype=np.float64)
    weights = weights / np.sum(weights)

    remaining_units = total_units - n_components
    ideal_extra_units = weights * remaining_units
    extra_units = np.floor(ideal_extra_units).astype(np.int64)
    missing_units = remaining_units - int(np.sum(extra_units))

    if missing_units > 0:
        remainders = ideal_extra_units - extra_units
        for idx in np.argsort(remainders)[-missing_units:]:
            extra_units[idx] += 1

    units = np.ones(n_components, dtype=np.int64) + extra_units
    return units_to_fractions(units, fraction_step)


def random_fraction_grid(
    n_components: int, fraction_step: float, rng: np.random.Generator
) -> np.ndarray:
    """Vytvoří náhodné složení po zvoleném kroku, bez nulových složek."""
    total_units = fraction_units(fraction_step)
    if n_components > total_units:
        raise ValueError(
            "Pocet slozek je prilis vysoky pro zvoleny krok podilu. "
            f"Pri kroku {fraction_step:.3f} lze mit maximalne {total_units} slozek."
        )

    remaining_units = total_units - n_components
    extra_units = rng.multinomial(
        remaining_units, np.full(n_components, 1.0 / n_components)
    )
    return units_to_fractions(np.ones(n_components, dtype=np.int64) + extra_units, fraction_step)


def make_cache_key(
    fluids: Sequence[str], fractions: Sequence[float], decimals: int
) -> Tuple:
    rounded = tuple(np.round(normalize_fractions(fractions), decimals).tolist())
    return tuple(fluids) + rounded


def penalty_evaluation(key: Tuple, error: str) -> Evaluation:
    """Vrátí neplatný bod s mírnou penalizací pro Pareto třídění."""
    return Evaluation(
        key=key,
        fitness=-PENALTY_VALUE,
        COP=-PENALTY_VALUE,
        VHC=-PENALTY_VALUE,
        W_comp=PENALTY_VALUE,
        error=error,
    )


def objective_values(evaluation: Evaluation) -> Tuple[float, float, float]:
    """Cíle převedené do směru maximalizace: COP, VHC, -W_comp."""
    return evaluation.COP, evaluation.VHC, -evaluation.W_comp


def dominates(evaluation_a: Evaluation, evaluation_b: Evaluation) -> bool:
    """True, pokud bod A Pareto-dominuje bod B."""
    objectives_a = objective_values(evaluation_a)
    objectives_b = objective_values(evaluation_b)
    return all(a >= b for a, b in zip(objectives_a, objectives_b)) and any(
        a > b for a, b in zip(objectives_a, objectives_b)
    )


def evaluate_mixture(
    fluids: Sequence[str], fractions: Sequence[float], params: CycleParams
) -> Evaluation:
    """Spočítá jeden konkrétní bod směsi přes CoolProp/REFPROP.

    Důležité: tato funkce sama neprochází žádnou mřížku podílů. Vždy dostane
    právě jednu kombinaci látek a právě jedno složení vytvořené GA.
    """
    import CoolProp.CoolProp as cp
    from CoolProp.CoolProp import AbstractState

    fractions = normalize_fractions(fractions)
    key = make_cache_key(fluids, fractions, decimals=10)

    if len(fluids) != len(fractions):
        return penalty_evaluation(key, "Pocet latek a podilu se neshoduje.")

    if len(set(fluids)) != len(fluids):
        return penalty_evaluation(key, "Smes obsahuje duplicitni latky.")

    try:
        AS = AbstractState(params.backend, "&".join(fluids))
        AS.set_mole_fractions(fractions.tolist())

        T1 = params.T_evap + params.dT_SH
        T3 = params.T_cond - params.dT_SC

        AS.specify_phase(cp.iphase_gas)
        AS.update(cp.QT_INPUTS, 1, params.T_evap)
        P1 = AS.p()

        AS.update(cp.PT_INPUTS, P1, T1)
        h1 = AS.hmass()
        s1 = AS.smass()
        rho1 = AS.rhomass()

        AS.specify_phase(cp.iphase_liquid)
        AS.update(cp.QT_INPUTS, 0, params.T_cond)
        P3 = AS.p()

        AS.update(cp.PT_INPUTS, P3, T3)
        h3 = AS.hmass()

        AS.unspecify_phase()
        AS.update(cp.PSmass_INPUTS, P3, s1)
        h2s = AS.hmass()

        h2 = h1 + (h2s - h1) / params.eta_comp
        AS.update(cp.HmassP_INPUTS, h2, P3)

        h4 = h3
        AS.update(cp.HmassP_INPUTS, h4, P1)

        q_in = h1 - h4
        q_out = h2 - h3
        w_cycle = q_out - q_in

        if q_out <= 0.0 or w_cycle <= 0.0:
            raise ValueError(
                f"Nefyzikalni cyklus: q_out={q_out:.6g}, w_cycle={w_cycle:.6g}"
            )

        m_dot = params.Q_out / q_out
        W_comp = w_cycle * m_dot
        COP = q_out / w_cycle
        VHC = q_out * rho1

        values = (COP, VHC, W_comp)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Vypocet vratil neplatnou hodnotu.")

        # fitness zde není optimalizační cíl. Je to jen pomocná hodnota pro
        # zpětnou kompatibilitu výstupu; GA vybírá podle Pareto ranku.
        return Evaluation(key, float(COP), float(COP), float(VHC), float(W_comp))
    except Exception as exc:
        try:
            AS.unspecify_phase()
        except Exception:
            pass
        return penalty_evaluation(key, str(exc))


def evaluate_worker(args: Tuple[Tuple[str, ...], np.ndarray, CycleParams, int]) -> Evaluation:
    fluids, fractions, params, cache_decimals = args
    result = evaluate_mixture(fluids, fractions, params)
    return Evaluation(
        make_cache_key(fluids, fractions, cache_decimals),
        result.fitness,
        result.COP,
        result.VHC,
        result.W_comp,
        result.error,
    )


def random_individual(
    available_fluids: Sequence[str],
    n_components: int,
    fraction_step: float,
    rng: np.random.Generator,
) -> Individual:
    """Inicializace jedince: náhodné látky bez opakování a diskrétní podíly."""
    fluids = tuple(rng.choice(available_fluids, size=n_components, replace=False).tolist())
    fractions = random_fraction_grid(n_components, fraction_step, rng)
    return Individual(fluids, fractions)


def individual_key(individual: Individual, cache_decimals: int) -> Tuple:
    """Jednoznačný klíč jedince pro cache i kontrolu duplicit."""
    return make_cache_key(individual.fluids, individual.fractions, cache_decimals)


def evaluation_for(
    individual: Individual, cache: Dict[Tuple, Evaluation], cache_decimals: int
) -> Evaluation:
    """Vrátí vyhodnocení jedince z cache."""
    return cache[individual_key(individual, cache_decimals)]


def fast_non_dominated_sort(
    population: Sequence[Individual],
    cache: Dict[Tuple, Evaluation],
    cache_decimals: int,
) -> List[List[int]]:
    """Rozdělí populaci do Pareto front. Fronta 0 je nejlepší nedominovaná."""
    domination_counts = [0 for _ in population]
    dominated_sets = [[] for _ in population]
    fronts: List[List[int]] = [[]]

    for i, individual_i in enumerate(population):
        eval_i = evaluation_for(individual_i, cache, cache_decimals)
        for j, individual_j in enumerate(population):
            if i == j:
                continue
            eval_j = evaluation_for(individual_j, cache, cache_decimals)
            if dominates(eval_i, eval_j):
                dominated_sets[i].append(j)
            elif dominates(eval_j, eval_i):
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            population[i].pareto_rank = 0
            fronts[0].append(i)

    rank = 0
    while fronts[rank]:
        next_front = []
        for i in fronts[rank]:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    population[j].pareto_rank = rank + 1
                    next_front.append(j)
        rank += 1
        fronts.append(next_front)

    return fronts[:-1]


def assign_crowding_distance(
    population: Sequence[Individual],
    front: Sequence[int],
    cache: Dict[Tuple, Evaluation],
    cache_decimals: int,
) -> None:
    """Spočítá crowding distance pro jednu Pareto frontu."""
    if not front:
        return

    for idx in front:
        population[idx].crowding_distance = 0.0

    if len(front) <= 2:
        for idx in front:
            population[idx].crowding_distance = float("inf")
        return

    for objective_index in range(3):
        sorted_front = sorted(
            front,
            key=lambda idx: objective_values(
                evaluation_for(population[idx], cache, cache_decimals)
            )[objective_index],
        )
        first = sorted_front[0]
        last = sorted_front[-1]
        population[first].crowding_distance = float("inf")
        population[last].crowding_distance = float("inf")

        min_value = objective_values(
            evaluation_for(population[first], cache, cache_decimals)
        )[objective_index]
        max_value = objective_values(
            evaluation_for(population[last], cache, cache_decimals)
        )[objective_index]
        span = max_value - min_value
        if span <= 0.0:
            continue

        for pos in range(1, len(sorted_front) - 1):
            previous_value = objective_values(
                evaluation_for(population[sorted_front[pos - 1]], cache, cache_decimals)
            )[objective_index]
            next_value = objective_values(
                evaluation_for(population[sorted_front[pos + 1]], cache, cache_decimals)
            )[objective_index]
            population[sorted_front[pos]].crowding_distance += (
                next_value - previous_value
            ) / span


def assign_pareto_scores(
    population: Sequence[Individual],
    cache: Dict[Tuple, Evaluation],
    cache_decimals: int,
) -> List[List[int]]:
    """Přiřadí všem jedincům Pareto rank a crowding distance."""
    fronts = fast_non_dominated_sort(population, cache, cache_decimals)
    for front in fronts:
        assign_crowding_distance(population, front, cache, cache_decimals)
    return fronts


def tournament_select(
    population: Sequence[Individual], tournament_size: int, rng: np.random.Generator
) -> Individual:
    """Turnajový výběr: lepší Pareto rank, při shodě větší crowding distance."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    return min(
        (population[i] for i in indices),
        key=lambda ind: (ind.pareto_rank, -ind.crowding_distance),
    )


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    fraction_step: float,
    rng: np.random.Generator,
) -> Individual:
    """Crossover: náhodně míchá látky z obou rodičů a k nim skládá podíly.

    Pokud je látka u obou rodičů, dostane kombinaci jejich podílů. Pokud je jen
    u jednoho, přenese se její rodičovský podíl. Na konci se vše zaokrouhlí na
    diskrétní krok fraction_step.
    """
    n_components = len(parent_a.fluids)
    fluid_pool = list(dict.fromkeys(parent_a.fluids + parent_b.fluids))
    rng.shuffle(fluid_pool)

    if len(fluid_pool) < n_components:
        fluids = fluid_pool
    else:
        fluids = fluid_pool[:n_components]

    fractions_a = dict(zip(parent_a.fluids, parent_a.fractions))
    fractions_b = dict(zip(parent_b.fluids, parent_b.fractions))
    alpha = rng.random()
    fractions = []
    for fluid in fluids:
        value_a = fractions_a.get(fluid, 0.0)
        value_b = fractions_b.get(fluid, 0.0)
        if value_a > 0.0 and value_b > 0.0:
            fractions.append(alpha * value_a + (1.0 - alpha) * value_b)
        else:
            fractions.append(max(value_a, value_b))

    fractions = quantize_fractions(fractions, len(fluids), fraction_step)
    return Individual(tuple(fluids), fractions)


def mutate(
    individual: Individual,
    available_fluids: Sequence[str],
    fraction_step: float,
    fraction_mutation_steps: int,
    fluid_mutation_rate: float,
    rng: np.random.Generator,
) -> Individual:
    """Mutace podílů a látek.

    Podíly se mění přesunem diskrétních dílků mezi složkami. Tím nikdy nevznikne
    složka s podílem 0 % a součet zůstane přesně 100 %.
    """
    fluids = list(individual.fluids)
    units = np.rint(individual.fractions / fraction_step).astype(np.int64)

    for _ in range(fraction_mutation_steps):
        donors = np.where(units > 1)[0]
        if len(donors) == 0:
            break
        donor = int(rng.choice(donors))
        receiver = int(rng.integers(0, len(units)))
        while receiver == donor:
            receiver = int(rng.integers(0, len(units)))
        units[donor] -= 1
        units[receiver] += 1

    fractions = units_to_fractions(units, fraction_step)

    if rng.random() < fluid_mutation_rate:
        idx = int(rng.integers(0, len(fluids)))
        candidates = [fluid for fluid in available_fluids if fluid not in fluids]
        if candidates:
            fluids[idx] = str(rng.choice(candidates))

    return Individual(tuple(fluids), fractions)


def evaluate_population(
    population: Sequence[Individual],
    params: CycleParams,
    cache: Dict[Tuple, Evaluation],
    cache_decimals: int,
    processes: int,
) -> int:
    """Vyhodnotí populaci, ale CoolProp volá jen pro směsi, které nejsou v cache."""
    missing = {}
    for individual in population:
        key = make_cache_key(individual.fluids, individual.fractions, cache_decimals)
        if key not in cache:
            missing[key] = (individual.fluids, individual.fractions.copy(), params, cache_decimals)

    if missing:
        if processes == 1:
            results = [evaluate_worker(args) for args in missing.values()]
        else:
            with Pool(processes=processes) as pool:
                results = pool.map(evaluate_worker, missing.values())

        for result in results:
            cache[result.key] = result

    for individual in population:
        key = make_cache_key(individual.fluids, individual.fractions, cache_decimals)
        individual.fitness = cache[key].fitness

    return len(missing)


def compromise_score(
    individual: Individual,
    candidates: Sequence[Individual],
    cache: Dict[Tuple, Evaluation],
    cache_decimals: int,
) -> float:
    """Pomocné skóre jen pro výběr reprezentanta Pareto fronty do výpisu.

    Optimalizace samotná toto číslo nepoužívá; slouží pouze k tomu, abychom
    vedle celé Pareto fronty uměli ukázat jednu rozumnou kompromisní směs.
    """
    objective_matrix = np.array(
        [
            objective_values(evaluation_for(candidate, cache, cache_decimals))
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    values = np.array(
        objective_values(evaluation_for(individual, cache, cache_decimals)),
        dtype=np.float64,
    )
    mins = np.min(objective_matrix, axis=0)
    maxs = np.max(objective_matrix, axis=0)
    spans = maxs - mins
    normalized = np.divide(
        values - mins,
        spans,
        out=np.full_like(values, 0.5),
        where=spans > 0.0,
    )
    return float(np.sum(normalized))


def select_representative(
    candidates: Sequence[Individual],
    cache: Dict[Tuple, Evaluation],
    cache_decimals: int,
) -> Individual:
    """Vybere jeden kompromisní bod z Pareto fronty pro stručný terminálový výpis."""
    return max(
        candidates,
        key=lambda individual: compromise_score(individual, candidates, cache, cache_decimals),
    )


def deduplicate_population(
    population: Sequence[Individual],
    available_fluids: Sequence[str],
    n_components: int,
    fraction_step: float,
    cache_decimals: int,
    population_size: int,
    rng: np.random.Generator,
) -> List[Individual]:
    """Odstraní duplicitní jedince a doplní populaci náhodnými novými body."""
    unique_population: List[Individual] = []
    seen = set()

    for individual in population:
        key = individual_key(individual, cache_decimals)
        if key in seen:
            continue
        seen.add(key)
        unique_population.append(individual)
        if len(unique_population) == population_size:
            return unique_population

    attempts = 0
    max_attempts = population_size * 100
    while len(unique_population) < population_size and attempts < max_attempts:
        attempts += 1
        immigrant = random_individual(available_fluids, n_components, fraction_step, rng)
        key = individual_key(immigrant, cache_decimals)
        if key in seen:
            continue
        seen.add(key)
        unique_population.append(immigrant)

    while len(unique_population) < population_size:
        unique_population.append(
            random_individual(available_fluids, n_components, fraction_step, rng)
        )

    return unique_population


def add_random_immigrants(
    population: List[Individual],
    available_fluids: Sequence[str],
    n_components: int,
    fraction_step: float,
    immigrant_rate: float,
    rng: np.random.Generator,
) -> None:
    """Nahradí část nejhorších potomků novými náhodnými jedinci."""
    immigrant_count = int(round(len(population) * immigrant_rate))
    if immigrant_count <= 0:
        return

    for idx in range(len(population) - immigrant_count, len(population)):
        population[idx] = random_individual(available_fluids, n_components, fraction_step, rng)


def run_ga(
    available_fluids: Sequence[str],
    n_components: int,
    params: CycleParams,
    population_size: int = 100,
    generations: int = 100,
    elite_size: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.35,
    fraction_step: float = FRACTION_STEP,
    fraction_mutation_steps: int = 1,
    fluid_mutation_rate: float = 0.25,
    random_immigrant_rate: float = 0.10,
    early_stopping_rounds: Optional[int] = 20,
    cache_decimals: int = 2,
    processes: Optional[int] = None,
    seed: Optional[int] = None,
    log_progress: bool = True,
) -> Tuple[Individual, List[Dict[str, float]], Dict[Tuple, Evaluation], List[Individual]]:
    """Spustí genetický algoritmus pro libovolný počet složek.

    Parametr n_components určuje, kolik látek bude každá směs obsahovat.
    Z příkazové řádky ho nastavíš pomocí --components, např. --components 7.
    """
    if n_components < 2:
        raise ValueError("Pocet slozek musi byt alespon 2.")
    if n_components > len(available_fluids):
        raise ValueError("Pocet slozek je vetsi nez pocet dostupnych latek.")
    if n_components > fraction_units(fraction_step):
        raise ValueError(
            "Pocet slozek je vetsi nez pocet nenulovych podilu dostupnych "
            f"pri kroku {fraction_step:.3f}."
        )
    if elite_size >= population_size:
        raise ValueError("elite_size musi byt mensi nez velikost populace.")
    if tournament_size > population_size:
        raise ValueError("tournament_size nesmi byt vetsi nez velikost populace.")

    rng = np.random.default_rng(seed)
    processes = processes or max(1, cpu_count() - 1)
    cache: Dict[Tuple, Evaluation] = {}
    history: List[Dict[str, float]] = []

    population = [
        random_individual(available_fluids, n_components, fraction_step, rng)
        for _ in range(population_size)
    ]

    best: Optional[Individual] = None
    pareto_front: List[Individual] = []
    best_generation = 0

    for generation in range(generations):
        new_evaluations = evaluate_population(
            population, params, cache, cache_decimals, processes
        )
        fronts = assign_pareto_scores(population, cache, cache_decimals)
        pareto_front = [
            Individual(
                population[idx].fluids,
                population[idx].fractions.copy(),
                population[idx].fitness,
                population[idx].pareto_rank,
                population[idx].crowding_distance,
            )
            for idx in fronts[0]
        ]
        population.sort(key=lambda ind: (ind.pareto_rank, -ind.crowding_distance))

        generation_best = select_representative(pareto_front, cache, cache_decimals)
        front_evaluations = [
            evaluation_for(individual, cache, cache_decimals) for individual in pareto_front
        ]
        best_cop = max(evaluation.COP for evaluation in front_evaluations)
        best_vhc = max(evaluation.VHC for evaluation in front_evaluations)
        best_w_comp = min(evaluation.W_comp for evaluation in front_evaluations)
        history.append(
            {
                "generation": generation,
                "pareto_front_size": float(len(pareto_front)),
                "best_COP": float(best_cop),
                "best_VHC": float(best_vhc),
                "best_W_comp": float(best_w_comp),
                "new_evaluations": float(new_evaluations),
                "cache_size": float(len(cache)),
            }
        )

        representative_score = compromise_score(
            generation_best, pareto_front, cache, cache_decimals
        )
        best_score = (
            -np.inf
            if best is None
            else compromise_score(best, pareto_front + [best], cache, cache_decimals)
        )
        if best is None or representative_score > best_score:
            best = Individual(
                generation_best.fluids,
                generation_best.fractions.copy(),
                generation_best.fitness,
                generation_best.pareto_rank,
                generation_best.crowding_distance,
            )
            best_generation = generation

        if log_progress:
            logging.info(
                (
                    "generace %d | Pareto fronta %d | max COP %.6g | "
                    "max VHC %.6g | min W_comp %.6g | nove vypocty %d | cache %d"
                ),
                generation,
                len(pareto_front),
                best_cop,
                best_vhc,
                best_w_comp,
                new_evaluations,
                len(cache),
            )

        if (
            early_stopping_rounds is not None
            and generation - best_generation >= early_stopping_rounds
        ):
            logging.info(
                "early stopping: bez zlepseni %d generaci", early_stopping_rounds
            )
            break

        elites = [
            Individual(
                ind.fluids,
                ind.fractions.copy(),
                ind.fitness,
                ind.pareto_rank,
                ind.crowding_distance,
            )
            for ind in population[:elite_size]
        ]
        next_population = elites

        while len(next_population) < population_size:
            parent_a = tournament_select(population, tournament_size, rng)
            parent_b = tournament_select(population, tournament_size, rng)
            child = crossover(parent_a, parent_b, fraction_step, rng)
            if rng.random() < mutation_rate:
                child = mutate(
                    child,
                    available_fluids,
                    fraction_step,
                    fraction_mutation_steps,
                    fluid_mutation_rate,
                    rng,
                )
            next_population.append(child)

        add_random_immigrants(
            next_population,
            available_fluids,
            n_components,
            fraction_step,
            random_immigrant_rate,
            rng,
        )
        population = deduplicate_population(
            next_population,
            available_fluids,
            n_components,
            fraction_step,
            cache_decimals,
            population_size,
            rng,
        )

    return best, history, cache, pareto_front


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimalizace slozeni chladivove smesi genetickym algoritmem. "
            "Pocet slozek nastav pres --components."
        )
    )
    parser.add_argument(
        "--fluids-file",
        default=os.path.join("směsi", "pure_fluids.txt"),
        help="Soubor se seznamem dostupnych cistych latek.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=komponenty,
        help="Pocet slozek ve smesi. Pro 5 az 7 slozek zadej napr. --components 7.",
    )
    parser.add_argument(
        "--fraction-step",
        type=float,
        default=FRACTION_STEP,
        help="Krok molarniho podilu. Hodnota 0.02 znamena podily po 2%%.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Velikost populace GA. Vetsi hodnota hleda lepe, ale pocita dele.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Maximalni pocet generaci GA.",
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=4,
        help="Pocet nejlepsich jedincu, kteri automaticky preziji do dalsi generace.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Pocet jedincu v turnajovem vyberu rodicu.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.35,
        help="Pravdepodobnost, ze potomek projde mutaci.",
    )
    parser.add_argument(
        "--fraction-mutation-steps",
        type=int,
        default=1,
        help="Kolik dilku podilu se pri mutaci presune mezi slozkami.",
    )
    parser.add_argument(
        "--fluid-mutation-rate",
        type=float,
        default=0.25,
        help="Pravdepodobnost, ze se pri mutaci vymeni jedna latka za jinou.",
    )
    parser.add_argument(
        "--random-immigrant-rate",
        type=float,
        default=0.10,
        help="Cast populace nahrazena nahodnymi jedinci kvuli diverzite.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=20,
        help="Zastavi GA, pokud se tolik generaci nezlepsi reprezentant Pareto fronty.",
    )
    parser.add_argument(
        "--cache-decimals",
        type=int,
        default=2,
        help="Zaokrouhleni podilu v cache klici. Pro 2%% krok staci 2.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Pocet procesu pro paralelni vyhodnocovani CoolProp.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Nahodne seminko pro opakovatelny beh.",
    )
    parser.add_argument(
        "--backend",
        default="REFPROP",
        help="CoolProp backend, typicky REFPROP nebo HEOS.",
    )
    parser.add_argument(
        "--pareto-file",
        default="pareto_front.csv",
        help="CSV soubor s finalni Pareto frontou.",
    )
    parser.add_argument("--T-evap", type=float, default=273.15 + 80.0, help="Teplota vyparu v K.")
    parser.add_argument("--T-cond", type=float, default=273.15 + 130.0, help="Teplota kondenzace v K.")
    parser.add_argument("--dT-SH", type=float, default=5.0, help="Prehrati v K.")
    parser.add_argument("--dT-SC", type=float, default=5.0, help="Podchlazeni v K.")
    parser.add_argument("--eta-comp", type=float, default=0.75, help="Izentropicka ucinnost kompresoru.")
    parser.add_argument("--Q-out", type=float, default=500000.0, help="Pozadovany tepelny vykon ve W.")
    parser.add_argument(
        "--history-file",
        default="ga_history.csv",
        help="CSV soubor pro historii Pareto fronty po generacich.",
    )
    parser.add_argument("--quiet", action="store_true", help="Vypne prubezne logovani.")
    return parser.parse_args()


def save_history(history: Iterable[Dict[str, float]], filename: str) -> None:
    rows = list(history)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    data = np.array(
        [[row[fieldname] for fieldname in fieldnames] for row in rows],
        dtype=np.float64,
    )
    np.savetxt(filename, data, delimiter=",", header=",".join(fieldnames), comments="")


def save_pareto_front(
    pareto_front: Sequence[Individual],
    cache: Dict[Tuple, Evaluation],
    cache_decimals: int,
    filename: str,
) -> None:
    """Uloží finální Pareto frontu do CSV pro další analýzu v Excelu/Pythonu."""
    if not pareto_front:
        return

    rows = []
    for individual in pareto_front:
        evaluation = evaluation_for(individual, cache, cache_decimals)
        rows.append(
            [
                "&".join(individual.fluids),
                ";".join(f"{fraction:.6f}" for fraction in individual.fractions),
                evaluation.COP,
                evaluation.VHC,
                evaluation.W_comp,
                individual.pareto_rank,
                individual.crowding_distance,
            ]
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("fluids,fractions,COP,VHC,W_comp,pareto_rank,crowding_distance\n")
        for row in rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.12g},{row[3]:.12g},"
                f"{row[4]:.12g},{row[5]},{row[6]:.12g}\n"
            )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    params = CycleParams(
        T_evap=args.T_evap,
        T_cond=args.T_cond,
        dT_SH=args.dT_SH,
        dT_SC=args.dT_SC,
        eta_comp=args.eta_comp,
        Q_out=args.Q_out,
        backend=args.backend,
    )

    available_fluids = load_fluids(args.fluids_file)
    best, history, cache, pareto_front = run_ga(
        available_fluids=available_fluids,
        n_components=args.components,
        params=params,
        population_size=args.population,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
        fraction_step=args.fraction_step,
        fraction_mutation_steps=args.fraction_mutation_steps,
        fluid_mutation_rate=args.fluid_mutation_rate,
        random_immigrant_rate=args.random_immigrant_rate,
        early_stopping_rounds=args.early_stopping,
        cache_decimals=args.cache_decimals,
        processes=args.processes,
        seed=args.seed,
        log_progress=not args.quiet,
    )

    best_key = make_cache_key(best.fluids, best.fractions, args.cache_decimals)
    best_eval = cache[best_key]
    save_history(history, args.history_file)
    save_pareto_front(pareto_front, cache, args.cache_decimals, args.pareto_file)

    print("\nReprezentativni kompromis z Pareto fronty")
    for fluid, fraction in zip(best.fluids, best.fractions):
        print(f"  {fluid:16s} {100.0 * fraction:6.2f} %")
    print(f"COP     = {best_eval.COP:.10g}")
    print(f"VHC     = {best_eval.VHC:.10g}")
    print(f"W_comp  = {best_eval.W_comp:.10g} W")
    print(f"velikost Pareto fronty = {len(pareto_front)}")
    print(f"historie Pareto vyvoje ulozena do: {args.history_file}")
    print(f"Pareto fronta ulozena do: {args.pareto_file}")


if __name__ == "__main__":
    main()
