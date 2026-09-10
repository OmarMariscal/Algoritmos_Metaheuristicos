"""
ga_continuo.py
===============
Motor de un Algoritmo Genético Continuo (representación real, punto flotante)
para minimización de funciones con dominio restringido tipo caja.

Arquitectura (mismo espíritu que la Actividad 2 - ga_binario.py):
    - GAConfig / ProblemSpec: dataclasses que centralizan la parametrización.
    - Sin "Encoding" binario: los individuos SON directamente vectores de
      floats (n_vars,), acotados a su dominio.
    - Operadores (selección, cruza, mutación, manejo de límites): Protocol +
      registry, intercambiables por nombre en GAConfig sin tocar el motor.
    - StoppingCriterion: paro fuerte (máx. generaciones), paro débil
      (estancamiento) y un criterio opcional de diversidad poblacional
      (desactivado por defecto), combinables.
    - Funciones objetivo: Ackley 2D (para comparar directo con la Actividad 2)
      y Esfera N-dimensional (parametrizable en número de variables).

Operadores continuos implementados (todos vectorizados con NumPy):
    Cruza:
        - Aritmética (whole arithmetic): combinación lineal convexa de
          ambos padres con un único alpha para todo el vector.
        - BLX-alpha (blend crossover): cada gen del hijo se muestrea
          uniformemente en un intervalo ampliado alrededor de los padres,
          lo que favorece la exploración (útil para evadir mínimos locales
          en funciones multimodales como Ackley).
        - Uniforme real: cada gen se hereda de un padre u otro (sin mezcla),
          análogo continuo del cruce uniforme discreto.
    Mutación:
        - Gaussiana: a cada gen, con probabilidad Pm, se le suma ruido
          N(0, sigma), con sigma proporcional al rango de esa variable.
        - Reinicio aleatorio (uniforme): con probabilidad Pm, el gen se
          reemplaza por un valor aleatorio uniforme dentro de su dominio
          (mutación "fuerte", útil para escapar de convergencia prematura).
    Manejo de límites tras cruza/mutación:
        - Clipping (default): recorte directo a la frontera del dominio.
        - Reflejo: "rebota" el valor excedente hacia adentro del dominio.
        - Resampling: si el valor cae fuera, se reemplaza por un valor
          aleatorio uniforme dentro del dominio.

Autor: Actividad 3 - Algoritmos Metaheurísticos (CUCEI, UdeG)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from numpy.random import Generator


# ======================================================================
# 1. Especificación del problema (función objetivo + dominio)
# ======================================================================

@dataclass(frozen=True)
class ProblemSpec:
    """Describe un problema de minimización con dominio tipo caja.

    Attributes:
        name: Identificador del problema (para reportes/registro).
        func: Función objetivo vectorizada: recibe (n_pop, n_vars) y
            devuelve (n_pop,).
        bounds: Límites [(min, max), ...] por variable.
        known_optimum: Valor óptimo global analítico, si se conoce.
        n_vars: Dimensión del problema, inferida de `bounds`.
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    bounds: Sequence[tuple[float, float]]
    known_optimum: float | None = None

    @property
    def n_vars(self) -> int:
        return len(self.bounds)


def _ackley(x: np.ndarray) -> np.ndarray:
    """Función de Ackley (2D): x, y en [-5, 5]. Óptimo global f(0,0) = 0.
    Misma función que en la Actividad 2, para comparación directa."""
    xv, yv = x[:, 0], x[:, 1]
    term1 = -20.0 * np.exp(-0.2 * np.sqrt((xv**2 + yv**2) / 2.0))
    term2 = -np.exp((np.cos(2 * np.pi * xv) + np.cos(2 * np.pi * yv)) / 2.0)
    return np.e + term1 + term2 + 20.0


def _sphere(x: np.ndarray) -> np.ndarray:
    """Función Esfera N-dimensional: f(x) = sum(xi^2). Óptimo global f(0,...,0) = 0."""
    return np.sum(x**2, axis=1)


def make_ackley_2d() -> ProblemSpec:
    """Construye el problema Ackley 2D, x, y en [-5, 5]."""
    return ProblemSpec(name="ackley_2d", func=_ackley, bounds=[(-5.0, 5.0), (-5.0, 5.0)], known_optimum=0.0)


def make_sphere(n_dims: int) -> ProblemSpec:
    """Construye el problema Esfera para `n_dims` variables, xi en [-5.12, 5.12]."""
    bounds = [(-5.12, 5.12)] * n_dims
    return ProblemSpec(name=f"sphere_{n_dims}d", func=_sphere, bounds=bounds, known_optimum=0.0)


# ======================================================================
# 2. Configuración del algoritmo
# ======================================================================

@dataclass
class GAConfig:
    """Hiperparámetros y opciones del GA continuo. Un único objeto que
    viaja por todo el motor, para facilitar cambios en vivo.

    Attributes:
        population_size: Tamaño de la población.
        max_generations: Criterio de paro "fuerte" (tope duro de iteraciones).
        crossover_prob: Probabilidad de cruce (Pc).
        mutation_prob: Probabilidad de mutación POR GEN (Pm).
        elitism: Número de mejores individuos que pasan sin alterar.
        selection: Operador de selección registrado en SELECTION_OPERATORS.
        tournament_k: Tamaño del torneo (si selection="tournament").
        crossover: Operador de cruza registrado en CROSSOVER_OPERATORS.
        crossover_alpha: Parámetro alpha usado por "arithmetic" y "blx_alpha".
        mutation: Operador de mutación registrado en MUTATION_OPERATORS.
        mutation_sigma_frac: Desviación estándar de la mutación gaussiana,
            como FRACCIÓN del rango [low, high] de cada variable (para que
            escale automáticamente con el dominio de cada problema).
        boundary_handling: Estrategia registrada en BOUNDARY_HANDLERS para
            corregir valores fuera de dominio tras cruza/mutación.
        stagnation_generations: Paro DÉBIL: generaciones sin mejora
            significativa antes de detenerse (0 = desactivado).
        stagnation_epsilon: Mejora mínima considerada significativa.
        diversity_epsilon: Criterio de paro OPCIONAL adicional: si la
            desviación estándar promedio de la población (normalizada por
            el rango de cada variable) cae por debajo de este umbral, se
            considera que la población colapsó (convergencia prematura).
            0.0 = desactivado (default), pensado como herramienta de
            diagnóstico/paro adicional configurable si se solicita en vivo.
        success_tolerance: Tolerancia |f_encontrado - f_óptimo| para éxito.
        seed: Semilla para reproducibilidad.
    """

    population_size: int = 60
    max_generations: int = 300
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    elitism: int = 2
    selection: str = "tournament"
    tournament_k: int = 3
    crossover: str = "blx_alpha"
    crossover_alpha: float = 0.5
    mutation: str = "gaussian"
    mutation_sigma_frac: float = 0.1
    boundary_handling: str = "clip"
    stagnation_generations: int = 30
    stagnation_epsilon: float = 1e-8
    diversity_epsilon: float = 0.0
    success_tolerance: float = 1e-3
    seed: int | None = None


# ======================================================================
# 3. Manejo de límites (dominio tipo caja)
# ======================================================================

class BoundaryHandler(Protocol):
    def __call__(self, values: np.ndarray, low: np.ndarray, high: np.ndarray, rng: Generator) -> np.ndarray:
        """Corrige `values` (n_pop, n_vars) para que respeten [low, high] por columna."""
        ...


def handle_clip(values: np.ndarray, low: np.ndarray, high: np.ndarray, rng: Generator) -> np.ndarray:
    """Recorte directo a la frontera del dominio (default)."""
    return np.clip(values, low, high)


def handle_reflect(values: np.ndarray, low: np.ndarray, high: np.ndarray, rng: Generator) -> np.ndarray:
    """Refleja el excedente hacia adentro del dominio (rebote), iterando por
    si el rebote inicial aún cae fuera (poco común pero posible)."""
    result = values.copy()
    for _ in range(3):  # suficiente en la práctica para rangos razonables
        below = result < low
        above = result > high
        if not (below.any() or above.any()):
            break
        result = np.where(below, 2 * low - result, result)
        result = np.where(above, 2 * high - result, result)
    return np.clip(result, low, high)  # salvaguarda final


def handle_resample(values: np.ndarray, low: np.ndarray, high: np.ndarray, rng: Generator) -> np.ndarray:
    """Sustituye cada valor fuera de dominio por un nuevo valor aleatorio
    uniforme dentro de [low, high] (mantiene diversidad, evita sesgo hacia
    la frontera que sí introduce el clipping)."""
    result = values.copy()
    out_of_bounds = (result < low) | (result > high)
    if out_of_bounds.any():
        random_vals = low + rng.random(values.shape) * (high - low)
        result = np.where(out_of_bounds, random_vals, result)
    return result


BOUNDARY_HANDLERS: dict[str, BoundaryHandler] = {
    "clip": handle_clip,
    "reflect": handle_reflect,
    "resample": handle_resample,
}


# ======================================================================
# 4. Operadores genéticos continuos (Protocol + registry -> intercambiables)
# ======================================================================

class SelectionOperator(Protocol):
    def __call__(
        self, population: np.ndarray, fitness: np.ndarray, n_select: int, rng: Generator, config: GAConfig
    ) -> np.ndarray:
        """Devuelve índices (n_select,) de individuos seleccionados como padres."""
        ...


class CrossoverOperator(Protocol):
    def __call__(
        self, parent_a: np.ndarray, parent_b: np.ndarray, rng: Generator, config: GAConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cruza dos vectores reales y devuelve dos hijos (mismo tamaño)."""
        ...


class MutationOperator(Protocol):
    def __call__(
        self, chromosome: np.ndarray, low: np.ndarray, high: np.ndarray, rng: Generator, config: GAConfig
    ) -> np.ndarray:
        """Aplica mutación a un vector real y devuelve el vector mutado
        (sin garantizar aún que respete [low, high]; eso lo hace el
        boundary handler después)."""
        ...


# --- Selección (representación-agnóstica: opera solo sobre fitness/índices) ---

def _fitness_from_minimization(raw_fitness: np.ndarray) -> np.ndarray:
    """Convierte f(x) (a minimizar) en aptitud positiva para ruleta."""
    worst = raw_fitness.max()
    return (worst - raw_fitness) + 1e-9


def select_roulette(
    population: np.ndarray, fitness: np.ndarray, n_select: int, rng: Generator, config: GAConfig
) -> np.ndarray:
    """Selección por ruleta (proporcional a la aptitud)."""
    adjusted = _fitness_from_minimization(fitness)
    probs = adjusted / adjusted.sum()
    return rng.choice(len(population), size=n_select, replace=True, p=probs)


def select_tournament(
    population: np.ndarray, fitness: np.ndarray, n_select: int, rng: Generator, config: GAConfig
) -> np.ndarray:
    """Selección por torneo de tamaño k (config.tournament_k)."""
    k = max(2, config.tournament_k)
    n_pop = len(population)
    selected = np.empty(n_select, dtype=np.int64)
    for i in range(n_select):
        contenders = rng.integers(0, n_pop, size=k)
        winner = contenders[np.argmin(fitness[contenders])]
        selected[i] = winner
    return selected


SELECTION_OPERATORS: dict[str, SelectionOperator] = {
    "roulette": select_roulette,
    "tournament": select_tournament,
}


# --- Cruza continua ---

def crossover_arithmetic(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: Generator, config: GAConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Cruza aritmética (whole arithmetic crossover):
        hijo1 = alpha * padre_a + (1 - alpha) * padre_b
        hijo2 = alpha * padre_b + (1 - alpha) * padre_a
    con un único `alpha` (config.crossover_alpha) aplicado a todo el vector.
    Genera siempre puntos intermedios entre los padres (buena explotación,
    poca exploración más allá del segmento que los une)."""
    if rng.random() >= config.crossover_prob:
        return parent_a.copy(), parent_b.copy()

    alpha = config.crossover_alpha
    child_a = alpha * parent_a + (1 - alpha) * parent_b
    child_b = alpha * parent_b + (1 - alpha) * parent_a
    return child_a, child_b


def crossover_blx_alpha(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: Generator, config: GAConfig
) -> tuple[np.ndarray, np.ndarray]:
    """BLX-alpha (blend crossover, Eshelman & Schaffer 1993).

    Para cada gen i, sea cmin = min(a_i, b_i), cmax = max(a_i, b_i),
    d = cmax - cmin. Cada gen del hijo se muestrea uniformemente en:
        [cmin - alpha*d, cmax + alpha*d]
    Es decir, no solo interpola entre los padres sino que EXTIENDE el
    intervalo un factor alpha hacia afuera, generando genes que pueden
    caer fuera del segmento que une a ambos padres. Esto introduce
    exploración adicional, lo que ayuda a evadir mínimos locales en
    funciones multimodales (ej. Ackley)."""
    if rng.random() >= config.crossover_prob:
        return parent_a.copy(), parent_b.copy()

    alpha = config.crossover_alpha
    cmin = np.minimum(parent_a, parent_b)
    cmax = np.maximum(parent_a, parent_b)
    d = cmax - cmin

    low = cmin - alpha * d
    high = cmax + alpha * d

    child_a = low + rng.random(len(parent_a)) * (high - low)
    child_b = low + rng.random(len(parent_a)) * (high - low)
    return child_a, child_b


def crossover_uniform_real(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: Generator, config: GAConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Cruza uniforme real: cada gen se hereda íntegro de un padre u otro
    (sin mezcla aritmética), con probabilidad 0.5 por gen. Análogo continuo
    directo del cruce uniforme discreto de la Actividad 2."""
    if rng.random() >= config.crossover_prob:
        return parent_a.copy(), parent_b.copy()

    mask = rng.integers(0, 2, size=len(parent_a)).astype(bool)
    child_a = np.where(mask, parent_a, parent_b)
    child_b = np.where(mask, parent_b, parent_a)
    return child_a, child_b


CROSSOVER_OPERATORS: dict[str, CrossoverOperator] = {
    "arithmetic": crossover_arithmetic,
    "blx_alpha": crossover_blx_alpha,
    "uniform_real": crossover_uniform_real,
}


# --- Mutación continua ---

def mutate_gaussian(
    chromosome: np.ndarray, low: np.ndarray, high: np.ndarray, rng: Generator, config: GAConfig
) -> np.ndarray:
    """Mutación gaussiana: a cada gen, independientemente, con probabilidad
    Pm (config.mutation_prob) se le suma ruido ~ N(0, sigma_i), donde
        sigma_i = config.mutation_sigma_frac * (high_i - low_i)
    es decir, la magnitud de la perturbación escala con el rango de cada
    variable (para que el mismo Pm/sigma_frac tenga un efecto comparable
    en dominios de distinto tamaño, ej. Ackley [-5,5] vs Esfera [-5.12,5.12])."""
    mutated = chromosome.copy()
    do_mutate = rng.random(len(chromosome)) < config.mutation_prob
    if not do_mutate.any():
        return mutated

    sigma = config.mutation_sigma_frac * (high - low)
    noise = rng.normal(loc=0.0, scale=sigma)
    mutated[do_mutate] = mutated[do_mutate] + noise[do_mutate]
    return mutated


def mutate_uniform_reset(
    chromosome: np.ndarray, low: np.ndarray, high: np.ndarray, rng: Generator, config: GAConfig
) -> np.ndarray:
    """Mutación por reinicio aleatorio: con probabilidad Pm, el gen se
    reemplaza por completo por un valor uniforme en [low, high]. Es una
    mutación "fuerte" (mayor exploración que la gaussiana), útil para
    inyectar diversidad y escapar de una convergencia prematura."""
    mutated = chromosome.copy()
    do_mutate = rng.random(len(chromosome)) < config.mutation_prob
    if not do_mutate.any():
        return mutated

    random_vals = low + rng.random(len(chromosome)) * (high - low)
    mutated[do_mutate] = random_vals[do_mutate]
    return mutated


MUTATION_OPERATORS: dict[str, MutationOperator] = {
    "gaussian": mutate_gaussian,
    "uniform_reset": mutate_uniform_reset,
}


# ======================================================================
# 5. Criterios de paro (combinables: fuerte + débil + diversidad opcional)
# ======================================================================

class StoppingCriterion(Protocol):
    def should_stop(self, generation: int, best_history: list[float], population: np.ndarray) -> bool:
        ...


@dataclass
class MaxGenerationsStopping:
    """Criterio de paro FUERTE: tope duro de generaciones."""

    max_generations: int

    def should_stop(self, generation: int, best_history: list[float], population: np.ndarray) -> bool:
        return generation >= self.max_generations


@dataclass
class StagnationStopping:
    """Criterio de paro DÉBIL: se detiene si el mejor fitness no mejora
    más que `epsilon` durante `patience` generaciones consecutivas."""

    patience: int
    epsilon: float

    def should_stop(self, generation: int, best_history: list[float], population: np.ndarray) -> bool:
        if self.patience <= 0 or len(best_history) <= self.patience:
            return False
        recent = best_history[-(self.patience + 1):]
        improvement = recent[0] - min(recent)
        return improvement < self.epsilon


@dataclass
class DiversityStopping:
    """Criterio OPCIONAL adicional: se detiene si la desviación estándar
    promedio de la población, normalizada por el rango de cada variable,
    cae por debajo de `epsilon` (población colapsada / convergencia
    prematura). Desactivado si epsilon <= 0."""

    epsilon: float
    bounds: np.ndarray  # (n_vars, 2)

    def should_stop(self, generation: int, best_history: list[float], population: np.ndarray) -> bool:
        if self.epsilon <= 0:
            return False
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        normalized_std = (population.std(axis=0) / ranges).mean()
        return bool(normalized_std < self.epsilon)


@dataclass
class CombinedStopping:
    """Combina varios criterios: se detiene si CUALQUIERA se cumple."""

    criteria: list[StoppingCriterion]

    def should_stop(self, generation: int, best_history: list[float], population: np.ndarray) -> bool:
        return any(c.should_stop(generation, best_history, population) for c in self.criteria)


def build_stopping_criterion(config: GAConfig, bounds: np.ndarray) -> CombinedStopping:
    """Arma el criterio de paro combinado a partir de GAConfig."""
    return CombinedStopping(
        criteria=[
            MaxGenerationsStopping(max_generations=config.max_generations),
            StagnationStopping(patience=config.stagnation_generations, epsilon=config.stagnation_epsilon),
            DiversityStopping(epsilon=config.diversity_epsilon, bounds=bounds),
        ]
    )


# ======================================================================
# 6. Resultado de una corrida
# ======================================================================

@dataclass
class RunResult:
    """Resultado de una corrida completa del GA continuo."""

    best_x: np.ndarray
    best_fitness: float
    generations_run: int
    convergence_best: list[float] = field(default_factory=list)
    convergence_mean: list[float] = field(default_factory=list)
    stopped_reason: str = ""


GenerationCallback = Callable[[int, np.ndarray, np.ndarray], None]


# ======================================================================
# 7. Motor evolutivo (agnóstico de los operadores concretos)
# ======================================================================

class GeneticAlgorithm:
    """Motor del Algoritmo Genético Continuo. Obtiene los operadores
    concretos de los registries según `config`, por lo que cambiar de
    operador (ej. en la evaluación presencial) es solo cambiar un campo
    de GAConfig, sin tocar esta clase."""

    def __init__(self, problem: ProblemSpec, config: GAConfig) -> None:
        self.problem = problem
        self.config = config
        self.low = np.array([b[0] for b in problem.bounds], dtype=np.float64)
        self.high = np.array([b[1] for b in problem.bounds], dtype=np.float64)
        self.bounds_arr = np.array(problem.bounds, dtype=np.float64)  # (n_vars, 2)

        self.selection_fn = self._lookup(SELECTION_OPERATORS, config.selection, "selección")
        self.crossover_fn = self._lookup(CROSSOVER_OPERATORS, config.crossover, "cruza")
        self.mutation_fn = self._lookup(MUTATION_OPERATORS, config.mutation, "mutación")
        self.boundary_fn = self._lookup(BOUNDARY_HANDLERS, config.boundary_handling, "manejo de límites")

    @staticmethod
    def _lookup(registry: dict, key: str, label: str):
        try:
            return registry[key]
        except KeyError as e:
            raise ValueError(f"Operador de {label} '{key}' no registrado. Disponibles: {list(registry)}") from e

    def _random_population(self, size: int, rng: Generator) -> np.ndarray:
        return self.low + rng.random((size, self.problem.n_vars)) * (self.high - self.low)

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        return self.problem.func(population)

    def run(self, rng: Generator | None = None, callback: GenerationCallback | None = None) -> RunResult:
        """Ejecuta el GA continuo completo. Ver GeneticAlgorithm.run en
        ga_binario.py para la contraparte binaria (misma filosofía)."""
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        stopping = build_stopping_criterion(self.config, self.bounds_arr)

        population = self._random_population(self.config.population_size, rng)
        fitness = self._evaluate(population)

        convergence_best: list[float] = []
        convergence_mean: list[float] = []

        generation = 0
        stopped_reason = "max_generations"

        while True:
            convergence_best.append(float(fitness.min()))
            convergence_mean.append(float(fitness.mean()))

            if callback is not None:
                callback(generation, population, fitness)

            if stopping.should_stop(generation, convergence_best, population):
                if generation >= self.config.max_generations:
                    stopped_reason = "max_generations"
                elif self.config.diversity_epsilon > 0 and self._diversity_collapsed(population):
                    stopped_reason = "diversity_collapse"
                else:
                    stopped_reason = "stagnation"
                break

            population, fitness = self._next_generation(population, fitness, rng)
            generation += 1

        best_idx = int(np.argmin(fitness))
        return RunResult(
            best_x=population[best_idx].copy(),
            best_fitness=float(fitness[best_idx]),
            generations_run=generation,
            convergence_best=convergence_best,
            convergence_mean=convergence_mean,
            stopped_reason=stopped_reason,
        )

    def _diversity_collapsed(self, population: np.ndarray) -> bool:
        ranges = self.bounds_arr[:, 1] - self.bounds_arr[:, 0]
        normalized_std = (population.std(axis=0) / ranges).mean()
        return bool(normalized_std < self.config.diversity_epsilon)

    def _next_generation(
        self, population: np.ndarray, fitness: np.ndarray, rng: Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n_pop = len(population)
        config = self.config

        # --- Elitismo ---
        elite_count = max(0, min(config.elitism, n_pop))
        elite_idx = np.argsort(fitness)[:elite_count]
        new_population = [population[i].copy() for i in elite_idx]

        # --- Reproducción ---
        while len(new_population) < n_pop:
            parents_idx = self.selection_fn(population, fitness, 2, rng, config)
            parent_a, parent_b = population[parents_idx[0]], population[parents_idx[1]]

            child_a, child_b = self.crossover_fn(parent_a, parent_b, rng, config)
            child_a = self.mutation_fn(child_a, self.low, self.high, rng, config)
            child_b = self.mutation_fn(child_b, self.low, self.high, rng, config)

            new_population.append(child_a)
            if len(new_population) < n_pop:
                new_population.append(child_b)

        new_population_arr = np.array(new_population, dtype=np.float64)
        new_population_arr = self.boundary_fn(new_population_arr, self.low, self.high, rng)
        new_fitness = self._evaluate(new_population_arr)
        return new_population_arr, new_fitness


# ======================================================================
# 8. Utilidad de éxito
# ======================================================================

def is_success(problem: ProblemSpec, best_fitness: float, tolerance: float) -> bool | None:
    """Determina si una corrida fue "exitosa" comparando contra el óptimo
    analítico conocido. None si el problema no tiene óptimo conocido."""
    if problem.known_optimum is None:
        return None
    return abs(best_fitness - problem.known_optimum) < tolerance


# ======================================================================
# 9. Utilidad para la discusión analítica del reporte
# ======================================================================

def binary_search_space_size(bits_per_var: int, n_vars: int) -> int:
    """Calcula el tamaño del espacio de búsqueda binario total
    (2 ** (bits_per_var * n_vars)), para la pregunta de discusión sobre
    codificación binaria vs continua en problemas multivariables."""
    return 2 ** (bits_per_var * n_vars)