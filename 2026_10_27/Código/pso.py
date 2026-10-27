"""
Autor: Mariscal Rodríguez Omar Jesús
Materia: Algoritmos Metaheurísticos
Profesor: Paredes López Ángel Ignasio
Actividad 4 - Algoritmo de Colonia de Abejas Artificiales

Universidad de Guadalajara
Centro Universitario de Ciencias Exactas e Ingenierías

------------------------------------------------------------------------



pso.py
======
Motor de Optimización por Enjambre de Partículas (Particle Swarm Optimization,
PSO) para minimización de funciones con dominio restringido tipo caja.

Ecuaciones de actualización (por partícula i, dimensión d):
    v_id <- w * v_id + c1 * r1 * (pbest_id - x_id) + c2 * r2 * (lbest_id - x_id)
    x_id <- x_id + v_id
donde r1, r2 ~ U(0, 1) se muestrean por partícula y por dimensión, `pbest` es la
mejor posición histórica de la partícula y `lbest` es la mejor posición
histórica (pbest) dentro de su VECINDAD (si la vecindad es todo el enjambre,
lbest = gbest).

Arquitectura (mismo espíritu que ga_binario.py, ga_continuo.py y abc_colony.py):
    - PSOConfig: dataclass que centraliza la parametrización.
    - ProblemSpec: función objetivo vectorizada + dominio + óptimo conocido.
    - Estrategias intercambiables vía Protocol + registry (se eligen por
      nombre en PSOConfig, sin tocar el motor):
        - NeighborhoodTopology (quién es la vecindad de cada partícula):
            "ring"          -> anillo: k vecinos por cada lado (lbest clásico).
            "global"        -> vecindad = todo el enjambre (gbest).
            "euclidean_knn" -> los k vecinos más cercanos en distancia
                               euclidiana en el espacio de búsqueda
                               (vecindad espacial dinámica).
        - InertiaSchedule (cómo evoluciona w durante la corrida):
            "constant"     -> w fijo.
            "linear_decay" -> decae linealmente de w_start a w_end.
        - BoundaryHandler (qué pasa cuando una partícula sale del dominio):
            "clip"         -> recorta la posición y anula la velocidad
                              en esa dimensión.
            "reflect"      -> refleja la posición e invierte la velocidad.
            "random_reset" -> reubica aleatoriamente y anula la velocidad.
    - Límite de velocidad v_max (fracción del rango de cada variable).
    - StoppingCriterion: paro fuerte (máx. iteraciones), paro débil
      (estancamiento del mejor global) y colapso de diversidad euclidiana
      (opcional, desactivado por defecto).
    - Se registra por iteración: mejor global, fitness medio del enjambre y
      diversidad euclidiana (distancia media al centroide, normalizada).

Autor: Actividad 5 - Algoritmos Metaheurísticos (CUCEI, UdeG)
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
        name: Identificador del problema.
        func: Función objetivo vectorizada: (n_pop, n_vars) -> (n_pop,).
        bounds: Límites [(min, max), ...] por variable.
        known_optimum: Valor óptimo global conocido (None si se desconoce).
        optimum_position: Posición del óptimo (solo para marcarlo en gráficas).
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    bounds: Sequence[tuple[float, float]]
    known_optimum: float | None = None
    optimum_position: tuple[float, ...] | None = None

    @property
    def n_vars(self) -> int:
        return len(self.bounds)


def _rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock: sum_i [(1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2].
    Para n=2: (1-x)^2 + 100 (y - x^2)^2. Óptimo global f(1,...,1) = 0 en un
    valle curvo y estrecho (difícil de recorrer: reto de explotación)."""
    return np.sum((1.0 - x[:, :-1]) ** 2 + 100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2, axis=1)


def make_rosenbrock(n_dims: int = 2) -> ProblemSpec:
    """Rosenbrock, x_i en [-5, 5]."""
    return ProblemSpec(
        name=f"rosenbrock_{n_dims}d",
        func=_rosenbrock,
        bounds=[(-5.0, 5.0)] * n_dims,
        known_optimum=0.0,
        optimum_position=(1.0,) * n_dims,
    )


def make_michalewicz(n_dims: int = 2, m: int = 10) -> ProblemSpec:
    """Michalewicz: f(x) = -sum_i sin(x_i) * sin^{2m}(i * x_i^2 / pi),
    x_i en [0, pi]. Muchos mínimos locales y crestas muy angostas para
    m grande (reto de exploración). Para n=2, m=10 el óptimo conocido es
    f ~= -1.801303 en (2.2029, 1.5708)."""
    idx = np.arange(1, n_dims + 1, dtype=np.float64)

    def _michalewicz(x: np.ndarray) -> np.ndarray:
        return -np.sum(np.sin(x) * np.sin(idx * x**2 / np.pi) ** (2 * m), axis=1)

    is_reference_case = (n_dims == 2 and m == 10)
    return ProblemSpec(
        name=f"michalewicz_{n_dims}d",
        func=_michalewicz,
        bounds=[(0.0, float(np.pi))] * n_dims,
        known_optimum=-1.801303410098553 if is_reference_case else None,
        optimum_position=(2.20290551, 1.57079633) if is_reference_case else None,
    )


# ======================================================================
# 2. Configuración del algoritmo
# ======================================================================

@dataclass
class PSOConfig:
    """Hiperparámetros y opciones del PSO.

    Attributes:
        swarm_size: Número de partículas (N).
        max_iterations: Criterio de paro FUERTE (tope duro de iteraciones).
        w: Peso de inercia (usado por el esquema "constant").
        c1: Coeficiente cognitivo (atracción hacia pbest).
        c2: Coeficiente social (atracción hacia lbest).
        topology: Estrategia de vecindad registrada en TOPOLOGIES.
        neighborhood_k: Tamaño de vecindad. "ring": vecinos por CADA lado
            (k=1 -> vecindad de 3 con la propia). "euclidean_knn": número de
            vecinos más cercanos (además de la propia partícula).
            Ignorado por "global".
        inertia_schedule: Esquema registrado en INERTIA_SCHEDULES.
        w_start, w_end: Extremos de "linear_decay".
        boundary_handling: Estrategia registrada en BOUNDARY_HANDLERS.
        v_max_frac: Límite de velocidad como fracción del rango de cada
            variable; también define el rango de velocidades iniciales
            (uniforme en [-v_max, v_max]).
        stagnation_iterations: Paro DÉBIL: iteraciones sin mejora
            significativa del mejor global (0 = desactivado).
        stagnation_epsilon: Mejora mínima considerada significativa.
        diversity_epsilon: Paro OPCIONAL por colapso de diversidad euclidiana
            normalizada (0.0 = desactivado).
        success_tolerance: Tolerancia |f_encontrado - f_óptimo| para éxito.
        seed: Semilla para reproducibilidad.
    """

    swarm_size: int = 40
    max_iterations: int = 300
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    topology: str = "ring"
    neighborhood_k: int = 1
    inertia_schedule: str = "constant"
    w_start: float = 0.9
    w_end: float = 0.4
    boundary_handling: str = "clip"
    v_max_frac: float = 0.2
    stagnation_iterations: int = 50
    stagnation_epsilon: float = 1e-10
    diversity_epsilon: float = 0.0
    success_tolerance: float = 1e-3
    seed: int | None = None


# ======================================================================
# 3. Estrategia: topología de vecindad (Protocol + registry)
# ======================================================================

class NeighborhoodTopology(Protocol):
    def __call__(
        self, positions: np.ndarray, pbest_pos: np.ndarray, pbest_fit: np.ndarray, config: PSOConfig
    ) -> np.ndarray:
        """Devuelve, para cada partícula, el ÍNDICE de la partícula cuyo
        pbest es el mejor de su vecindad (lbest). Forma: (n_particles,)."""
        ...


def topology_global(
    positions: np.ndarray, pbest_pos: np.ndarray, pbest_fit: np.ndarray, config: PSOConfig
) -> np.ndarray:
    """Vecindad = todo el enjambre (gbest). Convergencia rápida, mayor riesgo
    de convergencia prematura."""
    return np.full(len(pbest_fit), int(np.argmin(pbest_fit)), dtype=np.int64)


def topology_ring(
    positions: np.ndarray, pbest_pos: np.ndarray, pbest_fit: np.ndarray, config: PSOConfig
) -> np.ndarray:
    """Anillo (lbest clásico): la vecindad de la partícula i son las
    partículas i-k, ..., i+k (índices módulo N), incluida ella misma. La
    información se propaga lentamente por el anillo, preservando diversidad."""
    n = len(pbest_fit)
    k = max(1, config.neighborhood_k)
    offsets = np.arange(-k, k + 1)
    neighbor_idx = (np.arange(n)[:, None] + offsets[None, :]) % n      # (n, 2k+1)
    neighbor_fit = pbest_fit[neighbor_idx]                              # (n, 2k+1)
    best_local = np.argmin(neighbor_fit, axis=1)                        # (n,)
    return neighbor_idx[np.arange(n), best_local]


def topology_euclidean_knn(
    positions: np.ndarray, pbest_pos: np.ndarray, pbest_fit: np.ndarray, config: PSOConfig
) -> np.ndarray:
    """Vecindad espacial: la partícula i y sus k vecinos más cercanos en
    distancia euclidiana (calculada sobre las posiciones actuales). La
    vecindad cambia dinámicamente conforme el enjambre se mueve."""
    n = len(pbest_fit)
    k = min(max(1, config.neighborhood_k), n - 1)
    diffs = positions[:, None, :] - positions[None, :, :]               # (n, n, d)
    dists = np.linalg.norm(diffs, axis=2)                               # (n, n)
    # k+1 más cercanos (incluye a la propia partícula, a distancia 0)
    nearest = np.argsort(dists, axis=1)[:, : k + 1]                     # (n, k+1)
    neighbor_fit = pbest_fit[nearest]
    best_local = np.argmin(neighbor_fit, axis=1)
    return nearest[np.arange(n), best_local]


TOPOLOGIES: dict[str, NeighborhoodTopology] = {
    "ring": topology_ring,
    "global": topology_global,
    "euclidean_knn": topology_euclidean_knn,
}


# ======================================================================
# 4. Estrategia: esquema de inercia (Protocol + registry)
# ======================================================================

class InertiaSchedule(Protocol):
    def __call__(self, iteration: int, config: PSOConfig) -> float:
        """Devuelve el peso de inercia w a usar en la iteración dada."""
        ...


def inertia_constant(iteration: int, config: PSOConfig) -> float:
    """w constante (config.w)."""
    return config.w


def inertia_linear_decay(iteration: int, config: PSOConfig) -> float:
    """w decae linealmente de w_start a w_end a lo largo de max_iterations:
    w ALTO al inicio favorece exploración; w BAJO al final favorece
    explotación (afinar la solución)."""
    frac = min(1.0, iteration / max(1, config.max_iterations))
    return config.w_start + (config.w_end - config.w_start) * frac


INERTIA_SCHEDULES: dict[str, InertiaSchedule] = {
    "constant": inertia_constant,
    "linear_decay": inertia_linear_decay,
}


# ======================================================================
# 5. Estrategia: manejo de fronteras (Protocol + registry)
# ======================================================================

class BoundaryHandler(Protocol):
    def __call__(
        self, positions: np.ndarray, velocities: np.ndarray, low: np.ndarray, high: np.ndarray, rng: Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Corrige posiciones (y velocidades) para respetar [low, high]."""
        ...


def boundary_clip(positions, velocities, low, high, rng):
    """Recorta la posición a la frontera y anula la velocidad en las
    dimensiones donde se salió (evita que la partícula "empuje" el borde)."""
    out = (positions < low) | (positions > high)
    positions = np.clip(positions, low, high)
    velocities = np.where(out, 0.0, velocities)
    return positions, velocities


def boundary_reflect(positions, velocities, low, high, rng):
    """Refleja la posición hacia adentro e invierte la velocidad en las
    dimensiones donde se salió (rebote)."""
    below = positions < low
    above = positions > high
    positions = np.where(below, 2 * low - positions, positions)
    positions = np.where(above, 2 * high - positions, positions)
    velocities = np.where(below | above, -velocities, velocities)
    positions = np.clip(positions, low, high)  # salvaguarda por rebotes extremos
    return positions, velocities


def boundary_random_reset(positions, velocities, low, high, rng):
    """Reubica aleatoriamente (uniforme en el dominio) las coordenadas que
    se salieron y anula su velocidad (reinyecta diversidad)."""
    out = (positions < low) | (positions > high)
    if out.any():
        random_pos = low + rng.random(positions.shape) * (high - low)
        positions = np.where(out, random_pos, positions)
        velocities = np.where(out, 0.0, velocities)
    return positions, velocities


BOUNDARY_HANDLERS: dict[str, BoundaryHandler] = {
    "clip": boundary_clip,
    "reflect": boundary_reflect,
    "random_reset": boundary_random_reset,
}


# ======================================================================
# 6. Métrica de diversidad (distancia euclidiana)
# ======================================================================

def swarm_diversity_euclidean(positions: np.ndarray, bounds_arr: np.ndarray) -> float:
    """Distancia euclidiana MEDIA de las partículas al centroide del
    enjambre, normalizada por la diagonal del dominio (~0 = enjambre
    colapsado; mayor = más disperso)."""
    centroid = positions.mean(axis=0)
    mean_dist = float(np.linalg.norm(positions - centroid, axis=1).mean())
    diagonal = float(np.linalg.norm(bounds_arr[:, 1] - bounds_arr[:, 0]))
    return mean_dist / diagonal if diagonal > 0 else 0.0


# ======================================================================
# 7. Criterios de paro (combinables)
# ======================================================================

class StoppingCriterion(Protocol):
    def should_stop(self, iteration: int, best_history: list[float], positions: np.ndarray) -> bool:
        ...


@dataclass
class MaxIterationsStopping:
    """Paro FUERTE: tope duro de iteraciones."""

    max_iterations: int

    def should_stop(self, iteration: int, best_history: list[float], positions: np.ndarray) -> bool:
        return iteration >= self.max_iterations


@dataclass
class StagnationStopping:
    """Paro DÉBIL: el mejor global no mejora más que `epsilon` durante
    `patience` iteraciones consecutivas (0 = desactivado)."""

    patience: int
    epsilon: float

    def should_stop(self, iteration: int, best_history: list[float], positions: np.ndarray) -> bool:
        if self.patience <= 0 or len(best_history) <= self.patience:
            return False
        recent = best_history[-(self.patience + 1):]
        return (recent[0] - min(recent)) < self.epsilon


@dataclass
class DiversityStopping:
    """Paro OPCIONAL: diversidad euclidiana normalizada < epsilon."""

    epsilon: float
    bounds_arr: np.ndarray

    def should_stop(self, iteration: int, best_history: list[float], positions: np.ndarray) -> bool:
        if self.epsilon <= 0:
            return False
        return swarm_diversity_euclidean(positions, self.bounds_arr) < self.epsilon


@dataclass
class CombinedStopping:
    """Se detiene si CUALQUIERA de los criterios se cumple."""

    criteria: list[StoppingCriterion]

    def should_stop(self, iteration: int, best_history: list[float], positions: np.ndarray) -> bool:
        return any(c.should_stop(iteration, best_history, positions) for c in self.criteria)


def build_stopping_criterion(config: PSOConfig, bounds_arr: np.ndarray) -> CombinedStopping:
    return CombinedStopping(
        criteria=[
            MaxIterationsStopping(max_iterations=config.max_iterations),
            StagnationStopping(patience=config.stagnation_iterations, epsilon=config.stagnation_epsilon),
            DiversityStopping(epsilon=config.diversity_epsilon, bounds_arr=bounds_arr),
        ]
    )


# ======================================================================
# 8. Resultado de una corrida
# ======================================================================

@dataclass
class RunResult:
    """Resultado de una corrida completa del PSO."""

    best_x: np.ndarray
    best_fitness: float
    iterations_run: int
    convergence_best: list[float] = field(default_factory=list)   # mejor global por iteración
    convergence_mean: list[float] = field(default_factory=list)   # fitness medio del enjambre por iteración
    diversity_history: list[float] = field(default_factory=list)  # diversidad euclidiana por iteración
    stopped_reason: str = ""


IterationCallback = Callable[[int, np.ndarray, np.ndarray], None]


# ======================================================================
# 9. Motor PSO (agnóstico de las estrategias concretas)
# ======================================================================

class PSO:
    """Motor del PSO (actualización síncrona). Obtiene topología, esquema de
    inercia y manejo de fronteras de los registries según `config`, por lo
    que cambiarlos en vivo es solo cambiar un campo de PSOConfig."""

    def __init__(self, problem: ProblemSpec, config: PSOConfig) -> None:
        self.problem = problem
        self.config = config
        self.low = np.array([b[0] for b in problem.bounds], dtype=np.float64)
        self.high = np.array([b[1] for b in problem.bounds], dtype=np.float64)
        self.bounds_arr = np.array(problem.bounds, dtype=np.float64)
        self.v_max = config.v_max_frac * (self.high - self.low)

        self.topology_fn = self._lookup(TOPOLOGIES, config.topology, "topología")
        self.inertia_fn = self._lookup(INERTIA_SCHEDULES, config.inertia_schedule, "esquema de inercia")
        self.boundary_fn = self._lookup(BOUNDARY_HANDLERS, config.boundary_handling, "manejo de fronteras")

    @staticmethod
    def _lookup(registry: dict, key: str, label: str):
        try:
            return registry[key]
        except KeyError as e:
            raise ValueError(f"Estrategia de {label} '{key}' no registrada. Disponibles: {list(registry)}") from e

    def _evaluate(self, positions: np.ndarray) -> np.ndarray:
        return self.problem.func(positions)

    def run(self, rng: Generator | None = None, callback: IterationCallback | None = None) -> RunResult:
        """Ejecuta el PSO completo."""
        cfg = self.config
        rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        stopping = build_stopping_criterion(cfg, self.bounds_arr)

        n, d = cfg.swarm_size, self.problem.n_vars
        positions = self.low + rng.random((n, d)) * (self.high - self.low)
        velocities = rng.uniform(-self.v_max, self.v_max, size=(n, d))
        fitness = self._evaluate(positions)

        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()

        convergence_best: list[float] = []
        convergence_mean: list[float] = []
        diversity_history: list[float] = []

        iteration = 0
        stopped_reason = "max_iterations"

        while True:
            g_idx = int(np.argmin(pbest_fit))
            convergence_best.append(float(pbest_fit[g_idx]))
            convergence_mean.append(float(fitness.mean()))
            diversity_history.append(swarm_diversity_euclidean(positions, self.bounds_arr))

            if callback is not None:
                callback(iteration, positions, fitness)

            if stopping.should_stop(iteration, convergence_best, positions):
                if iteration >= cfg.max_iterations:
                    stopped_reason = "max_iterations"
                elif cfg.diversity_epsilon > 0 and diversity_history[-1] < cfg.diversity_epsilon:
                    stopped_reason = "diversity_collapse"
                else:
                    stopped_reason = "stagnation"
                break

            # --- Actualización síncrona de velocidad y posición ---
            w = self.inertia_fn(iteration, cfg)
            lbest_idx = self.topology_fn(positions, pbest_pos, pbest_fit, cfg)
            lbest_pos = pbest_pos[lbest_idx]

            r1 = rng.random((n, d))
            r2 = rng.random((n, d))
            velocities = (
                w * velocities
                + cfg.c1 * r1 * (pbest_pos - positions)
                + cfg.c2 * r2 * (lbest_pos - positions)
            )
            velocities = np.clip(velocities, -self.v_max, self.v_max)
            positions = positions + velocities
            positions, velocities = self.boundary_fn(positions, velocities, self.low, self.high, rng)

            fitness = self._evaluate(positions)
            improved = fitness < pbest_fit
            pbest_pos[improved] = positions[improved]
            pbest_fit[improved] = fitness[improved]

            iteration += 1

        g_idx = int(np.argmin(pbest_fit))
        return RunResult(
            best_x=pbest_pos[g_idx].copy(),
            best_fitness=float(pbest_fit[g_idx]),
            iterations_run=iteration,
            convergence_best=convergence_best,
            convergence_mean=convergence_mean,
            diversity_history=diversity_history,
            stopped_reason=stopped_reason,
        )


# ======================================================================
# 10. Utilidad de éxito
# ======================================================================

def is_success(problem: ProblemSpec, best_fitness: float, tolerance: float) -> bool | None:
    """Éxito si |f_encontrado - f_óptimo| < tolerancia. None si no hay óptimo conocido."""
    if problem.known_optimum is None:
        return None
    return abs(best_fitness - problem.known_optimum) < tolerance
