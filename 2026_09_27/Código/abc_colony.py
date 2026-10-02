"""

Autor: Mariscal Rodríguez Omar Jesús
Materia: Algoritmos Metaheurísticos
Profesor: Paredes López Ángel Ignasio
Actividad 4 - Algoritmo de Colonia de Abejas Artificiales

Universidad de Guadalajara
Centro Universitario de Ciencias Exactas e Ingenierías

------------------------------------------------------------------------

Motor del Algoritmo de Colonia de Abejas Artificiales (Artificial Bee Colony,
ABC) para minimización de funciones con dominio restringido tipo caja.

Arquitectura (mismo espíritu que ga_binario.py / ga_continuo.py de las
actividades previas):
    - GAConfig pasa a ser aquí ABCConfig: dataclass que centraliza la parametrización.
    - Representación real (fuentes de alimento = vectores de floats), sin
      codificación adicional.
    - Operadores intercambiables vía Protocol + registry dado el patrón de diseño de estrategia:
        - NeighborSelectionOperator: cómo se elige el vecino k en la
          búsqueda vecinal v_ij = x_ij + phi_ij*(x_ij - x_kj
              "random": ABC clásico: k aleatorio != i. Este es el visto en clase
              "nearest_euclidean": variante econtrada: k = vecino más cercano en
                                      distancia euclidiana (mayor explotación
                                      local, menor exploración que random).
        - ScoutStrategy: qué fuentes se abandonan/reinician cuando superan
          `limit` intentos sin mejorar.
              "single_worst": ABC clásico: solo la fuente más agotada
                                  que supere `limit`, una por ciclo.
              "all_exceeding": variante: Todas las fuentes que superen
                                  `limit` se reinician en el mismo ciclo.
    - StoppingCriterion: paro fuerte (máx. ciclos), paro débil (estancamiento
      del mejor histórico) y un criterio opcional de colapso de diversidad
      (basado en distancia euclidiana media al centro), desactivado por
      defecto.
    - Se registra en cada ciclo, además de la convergencia, la diversidad
      poblacional (distancia euclidiana media de las fuentes al centroide,
      normalizada por la diagonal del dominio) — información extra útil
      para ilustrar cómo las exploradoras reinyectan diversidad.

Las 3 fases del ciclo ABC son cada una de los tipos de abejas vistas:
    1. Abejas Empleadas: cada una de las SN fuentes intenta una búsqueda
       vecinal (greedy: se queda con el mejor entre la fuente actual y la
       candidata).
    2. Abejas Observadores: se seleccionan SN fuentes (con reemplazo) por
       RULETA proporcional a su aptitud, y sobre cada una se repite la
       misma búsqueda vecinal greedy. Esto concentra el esfuerzo de
       búsqueda en las regiones más prometedoras (explotación).
    3. Abejas Exploradoras: las fuentes que acumulan `trial >= limit`
       intentos sin mejorar se abandonan y se reinician aleatoriamente
       (diversificación / escape de mínimos locales).


"""

# Importaciones necesarias similares a las actividades anteriores
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from numpy.random import Generator

"""
1. Especificación del problema (función objetivo + dominio)
____________________________________________________________
Misma idea y contratos de las actividades pasadas
"""

@dataclass(frozen=True)
class ProblemSpec:
    """Describe un problema de minimización con dominio tipo caja."""

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    bounds: Sequence[tuple[float, float]]
    known_optimum: float | None = None

    @property
    def n_vars(self) -> int:
        return len(self.bounds)


def _eggholder(x: np.ndarray) -> np.ndarray:
    """Función Eggholder (2D), x, y en [-512, 512].
    Óptimo global conocido: f(512, 404.2319) ~= -959.6407.
    Superficie no convexa, extremadamente multimodal."""
    xv, yv = x[:, 0], x[:, 1]
    term1 = -(yv + 47) * np.sin(np.sqrt(np.abs(xv / 2.0 + (yv + 47))))
    term2 = -xv * np.sin(np.sqrt(np.abs(xv - (yv + 47))))
    return term1 + term2


def make_eggholder() -> ProblemSpec:
    """Construye el problema Eggholder, x, y en [-512, 512]."""
    return ProblemSpec(
        name="eggholder",
        func=_eggholder,
        bounds=[(-512.0, 512.0), (-512.0, 512.0)],
        known_optimum=-959.6407,
    )



"""
# 2. Configuración del algoritmo
____________________________________________________________
"""

@dataclass
class ABCConfig:
    """Hiperparámetros y opciones del ABC.

    Attributes:
        colony_size: Número de fuentes de alimento  o potenciales soluciones (SN). También el número
            de abejas empleadas (una por fuente).
        onlooker_count: Número de abejas observadoras. Si es None, se usa
            colony_size (proporción 1:1 clásica).
        limit: Número de intentos sin mejorar antes de abandonar una fuente y llamar a una abeja exploradora.
            Si es None, se calcula como colony_size * n_vars (regla común
            limit = SN x D).
        max_cycles: Criterio de paro FUERTE (tope duro de ciclos).
        neighbor_selection: Operador registrado en NEIGHBOR_SELECTION_OPERATORS
            para elegir el vecino k en la búsqueda vecinal.
        scout_strategy: Operador registrado en SCOUT_STRATEGIES para decidir
            qué fuentes se abandonan cuando superan `limit`.
        stagnation_cycles: Paro DÉBIL: ciclos sin mejora significativa del
            mejor histórico antes de detenerse (0 = desactivado).
        stagnation_epsilon: Mejora mínima considerada significativa.
        diversity_epsilon: Criterio de paro OPCIONAL adicional: si la
            diversidad euclidiana normalizada de la población cae por
            debajo de este umbral, se considera colapso de diversidad.
            0.0 = desactivado (default).
        success_tolerance: Tolerancia |f_encontrado - f_óptimo| para éxito.
            Dado el rango amplio y la rugosidad de Eggholder, 1.0 es un
            default razonable (ajustable).
        seed: Semilla para reproducibilidad.
    """

    colony_size: int = 40
    onlooker_count: int | None = None
    limit: int | None = None
    max_cycles: int = 500
    neighbor_selection: str = "random"
    scout_strategy: str = "single_worst"
    stagnation_cycles: int = 60
    stagnation_epsilon: float = 1e-6
    diversity_epsilon: float = 0.0
    success_tolerance: float = 1.0
    seed: int | None = None


"""
3. Operadores: selección de vecino (Protocol + registry)
____________________________________________________________
"""

# Protocolo de la selección de vencidad
class NeighborSelectionOperator(Protocol):
    def __call__(
        self, population: np.ndarray, index_i: int, rng: Generator, config: ABCConfig
    ) -> int:
        """Devuelve el índice k (!= index_i, es decir, distinto de él mismo) del vecino a usar en la
        búsqueda vecinal v_ij = x_ij + phi_ij*(x_ij - x_kj)."""
        ...

def select_neighbor_random(
    population: np.ndarray, index_i: int, rng: Generator, config: ABCConfig
) -> int:
    """ABC clásico: vecino aleatorio distinto de index_i (mayor exploración,
    ya que la dirección de búsqueda no está sesgada hacia soluciones
    similares)."""
    n = len(population)
    if n <= 1: # Protección en caso de que la población sea igual a 1. No hay vecinos
        return index_i
    k = index_i
    while k == index_i:
        k = int(rng.integers(0, n)) # Selección aleatoria
    return k

# Selección por distancia euclidiana
def select_neighbor_nearest_euclidean(
    population: np.ndarray, index_i: int, rng: Generator, config: ABCConfig
) -> int:
    """Variante: vecino más cercano en distancia euclidiana (excluyendo la
    propia fuente). Al perturbar hacia una solución similar, el paso
    (x_ij - x_kj) tiende a ser pequeño, favoreciendo una búsqueda más fina
    y local (más explotación, menos exploración que la variante aleatoria).
    
    El riesgo es no acabar de explorar el espacio de búsqueda
    """
    diffs = population - population[index_i]
    dists = np.linalg.norm(diffs, axis=1)
    dists[index_i] = np.inf  # excluir la propia fuente
    return int(np.argmin(dists))

# Listado de estrategias de la selección de vecinos
NEIGHBOR_SELECTION_OPERATORS: dict[str, NeighborSelectionOperator] = {
    "random": select_neighbor_random,
    "nearest_euclidean": select_neighbor_nearest_euclidean,
}

"""
4. Operadores: estrategia de exploradoras (Protocol + registry)
____________________________________________________________
"""

# Contraqto de la estrategia de selección de las exploradoras en cuestión de cuantas fuentes se pueden abandonar en cada ciclo si se supera el límite
class ScoutStrategy(Protocol):
    def __call__(self, trial_counts: np.ndarray, limit: int, rng: Generator) -> list[int]:
        """Devuelve los índices de fuentes a abandonar/reiniciar este ciclo."""
        ...


def scout_single_worst(trial_counts: np.ndarray, limit: int, rng: Generator) -> list[int]:
    """ABC clásico (Karaboga): como máximo UNA fuente por ciclo — la de
    mayor `trial` — se abandona, y solo si supera `limit`."""
    idx = int(np.argmax(trial_counts))
    if trial_counts[idx] >= limit:
        return [idx]
    return []


def scout_all_exceeding(trial_counts: np.ndarray, limit: int, rng: Generator) -> list[int]:
    """Variante: TODAS las fuentes que superen `limit` se abandonan en el
    mismo ciclo (diversificación más agresiva, útil si se busca escapar
    más rápido de zonas de estancamiento)."""
    return [int(i) for i in np.where(trial_counts >= limit)[0]]

# Listado de estrategias de las abejas exploradoras
SCOUT_STRATEGIES: dict[str, ScoutStrategy] = {
    "single_worst": scout_single_worst,
    "all_exceeding": scout_all_exceeding,
}


"""
5. Métrica de diversidad poblacional (distancia euclidiana) 
    Esto para evitar que todas las abejas se concentren en una fuente de alimento muy cerca
____________________________________________________________
"""

def population_diversity_euclidean(population: np.ndarray, bounds_arr: np.ndarray) -> float:
    """Diversidad poblacional: distancia euclidiana MEDIA de las fuentes
    al centroide de la población, normalizada por la diagonal del dominio
    (para obtener un valor en una escala interpretable, ~0 = colapsada,
    valores mayores = más dispersa), independiente de la escala del
    problema."""
    centroid = population.mean(axis=0)
    dists = np.linalg.norm(population - centroid, axis=1)
    mean_dist = float(dists.mean())
    domain_diagonal = float(np.linalg.norm(bounds_arr[:, 1] - bounds_arr[:, 0]))
    return mean_dist / domain_diagonal if domain_diagonal > 0 else 0.0



"""
6. Criterios de paro (combinables: fuerte + débil + diversidad opcional)
    Mista idea que con las actividades anteriores + diversidad opcional
____________________________________________________________
"""

#Protocolo del criterio de Paro
class StoppingCriterion(Protocol):
    def should_stop(self, cycle: int, best_history: list[float], population: np.ndarray) -> bool:
        ...


@dataclass
class MaxCyclesStopping:
    """Criterio de paro FUERTE: tope duro de ciclos."""

    max_cycles: int

    def should_stop(self, cycle: int, best_history: list[float], population: np.ndarray) -> bool:
        return cycle >= self.max_cycles


@dataclass
class StagnationStopping:
    """Criterio de paro DÉBIL: se detiene si el mejor histórico no mejora
    más que `epsilon` durante `patience` ciclos consecutivos."""

    patience: int
    epsilon: float

    def should_stop(self, cycle: int, best_history: list[float], population: np.ndarray) -> bool:
        if self.patience <= 0 or len(best_history) <= self.patience:
            return False
        recent = best_history[-(self.patience + 1):]
        improvement = recent[0] - min(recent)
        return improvement < self.epsilon


@dataclass
class DiversityStopping:
    """Criterio OPCIONAL: se detiene si la diversidad euclidiana normalizada
    de la población cae por debajo de `epsilon` (colapso de diversidad).
    Desactivado si epsilon <= 0.
    Usamos la función creada en la sección anterior

    """

    epsilon: float
    bounds_arr: np.ndarray

    def should_stop(self, cycle: int, best_history: list[float], population: np.ndarray) -> bool:
        if self.epsilon <= 0:
            return False
        return population_diversity_euclidean(population, self.bounds_arr) < self.epsilon


@dataclass
class CombinedStopping:
    """Combina varios criterios: se detiene si CUALQUIERA se cumple."""

    criteria: list[StoppingCriterion]

    def should_stop(self, cycle: int, best_history: list[float], population: np.ndarray) -> bool:
        return any(c.should_stop(cycle, best_history, population) for c in self.criteria)


def build_stopping_criterion(config: ABCConfig, bounds_arr: np.ndarray) -> CombinedStopping:
    return CombinedStopping(
        criteria=[
            MaxCyclesStopping(max_cycles=config.max_cycles), # Criterio de Paro Fuerte
            StagnationStopping(patience=config.stagnation_cycles, epsilon=config.stagnation_epsilon), # Criterio de Paro Débil
            DiversityStopping(epsilon=config.diversity_epsilon, bounds_arr=bounds_arr), # Criterio de Paro Extra para la Diversidad Poblacional
        ]
    )


"""
7. Resultado de una corrida
____________________________________________________________
Misma idea que las actividades pasadas, un contenedor de datos
"""

@dataclass
class RunResult:
    """Resultado de una corrida completa del ABC."""

    best_x: np.ndarray
    best_fitness: float
    cycles_run: int
    convergence_best: list[float] = field(default_factory=list)   # mejor histórico por ciclo
    diversity_history: list[float] = field(default_factory=list)  # diversidad euclidiana por ciclo
    stopped_reason: str = ""


CycleCallback = Callable[[int, np.ndarray, np.ndarray], None]

"""
8. Motor ABC (agnóstico de los operadores concretos)
____________________________________________________________
"""

class ABCAlgorithm:
    """Motor del Algoritmo de Colonia de Abejas Artificiales. Obtiene los
    operadores concretos (selección de vecino, estrategia de exploradoras)
    de los registries según `config`, por lo queun cambio es solo
    cambiar un campo de ABCConfig, sin tocar esta clase."""

    def __init__(self, problem: ProblemSpec, config: ABCConfig) -> None:
        self.problem = problem # Problema a Resolver
        self.config = config # Configuraciones
        self.low = np.array([b[0] for b in problem.bounds], dtype=np.float64) # Los límites bajos de cada dimensión
        self.high = np.array([b[1] for b in problem.bounds], dtype=np.float64) # Los límites altos de cada dimensión
        self.bounds_arr = np.array(problem.bounds, dtype=np.float64) # Arreglo de Límites
        self.n_vars = problem.n_vars # Número de variables del problema

        self.colony_size = config.colony_size # Tamaño de la colonia SN (Cantidad de Abejas Empleadas)
        self.onlooker_count = config.onlooker_count if config.onlooker_count is not None else config.colony_size # Cantidad de abejas observadoras según la estrategia seleccionada
        self.limit = config.limit if config.limit is not None else config.colony_size * self.n_vars # Límite de estancamiento de una fuente de alimento

        self.neighbor_fn = self._lookup(NEIGHBOR_SELECTION_OPERATORS, config.neighbor_selection, "selección de vecino") # Función para seleccionar al vecino
        self.scout_fn = self._lookup(SCOUT_STRATEGIES, config.scout_strategy, "estrategia de exploradoras") # Función para las abejas exploradoras

    # Función auxiliar para validar que las estrategias existan en sus listas
    @staticmethod
    def _lookup(registry: dict, key: str, label: str):
        try:
            return registry[key]
        except KeyError as e:
            raise ValueError(f"Operador de {label} '{key}' no registrado. Disponibles: {list(registry)}") from e

    # Generación aleatoria de las fuentes de alimento
    def _random_population(self, size: int, rng: Generator) -> np.ndarray:
        return self.low + rng.random((size, self.n_vars)) * (self.high - self.low)

    # Evaluación del fitness de la población de fuentes
    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        return self.problem.func(population)

    def _neighbor_search_candidate(self, population: np.ndarray, index_i: int, rng: Generator) -> np.ndarray:
        """Genera una solución candidata a partir de la fuente `index_i`
        mediante la fórmula vecinal v_ij = x_ij + phi_ij*(x_ij - x_kj),
        modificando una única dimensión j elegida al azar (esquema clásico
        de ABC, que perturba un componente a la vez).
        
        Es la exploración de los bordes de una fuente de alimento que se vio en clase
        """
        k = self.neighbor_fn(population, index_i, rng, self.config) # Selección del vecino
        j = int(rng.integers(0, self.n_vars)) # Selección de una dimensión al azar
        phi = rng.uniform(-1.0, 1.0) # Número aleatorio phi pertence a [-1,1]

        candidate = population[index_i].copy() # Fuente de alimento original
        candidate[j] = population[index_i, j] + phi * (population[index_i, j] - population[k, j]) # Modificación de la dimensión en base a la fórmula vecindad
        return np.clip(candidate, self.low, self.high) # En caso de que se salga del espacio de búsqueda, se corta a los bordes

    def _greedy_update(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        trial_counts: np.ndarray,
        index_i: int,
        rng: Generator,
    ) -> None:
        """Genera un candidato vecino a `index_i` y lo adopta (selección
        greedy) si mejora la fuente actual; si no, incrementa su contador
        de intentos fallidos (`trial`)."""
        candidate = self._neighbor_search_candidate(population, index_i, rng) # Encontrar una posible nueve fuente de alimento
        f_candidate = float(self.problem.func(candidate.reshape(1, -1))[0]) # Evaluación del fitness de la nueva fuente

        if f_candidate < fitness[index_i]: # Si la fuente candidata es mejor (Menor Fitness)
            population[index_i] = candidate # Actualizamos la fuente
            fitness[index_i] = f_candidate # Actualizamos el fitness de la fuente
            trial_counts[index_i] = 0 # Reiniciamos el contador de estancamiento
        else: # Si no fue mejor
            trial_counts[index_i] += 1 # Aumentamos el contador de estancamiento

    def _employed_phase(
        self, population: np.ndarray, fitness: np.ndarray, trial_counts: np.ndarray, rng: Generator
    ) -> None:
        """Fase de abejas Empleadas: cada fuente intenta una búsqueda
        vecinal (una abeja empleada por fuente)."""
        for i in range(self.colony_size): # Cada abeja en la colonia
            self._greedy_update(population, fitness, trial_counts, i, rng) # Intenta buscar una mejor fuente de alimento, puede conseguirlo o aumentar su contador de estancamiento

    # Selección de probabilidades que usarán las abejas observadoras por método de ruleta
    def _selection_probabilities(self, fitness: np.ndarray) -> np.ndarray:
        """Transforma f(x) (a minimizar) en probabilidades de selección
        para la ruleta de observadoras, con la fórmula estándar de ABC:
            fit(x) = 1 / (1 + f(x))      si f(x) >= 0
            fit(x) = 1 + |f(x)|          si f(x) < 0
        (garantiza aptitud positiva y monótona decreciente en f, incluso
        con valores de f negativos como en Eggholder)."""
        fit = np.where(fitness >= 0, 1.0 / (1.0 + fitness), 1.0 + np.abs(fitness))
        return fit / fit.sum()

    def _onlooker_phase(
        self, population: np.ndarray, fitness: np.ndarray, trial_counts: np.ndarray, rng: Generator
    ) -> None:
        """Fase de abejas Observadoras: se eligen `onlooker_count` fuentes
        (con reemplazo) por ruleta proporcional a su aptitud, y sobre cada
        una se repite la búsqueda vecinal greedy (concentra el esfuerzo de
        búsqueda en las zonas más prometedoras)."""
        probs = self._selection_probabilities(fitness) # Cálculo de probabilidades para la ruleta
        chosen = rng.choice(self.colony_size, size=self.onlooker_count, replace=True, p=probs) # Selección aleatoria de onlooker_count fuentes pudiéndose repetir
        for i in chosen:
            self._greedy_update(population, fitness, trial_counts, int(i), rng) # Cada observadora intenta buscar una fuente de alimento mejor dada en referencia a una fuente de alimento vecino
            # Comportamiento idéntico a la empleadas

    def _scout_phase(
        self, population: np.ndarray, fitness: np.ndarray, trial_counts: np.ndarray, rng: Generator
    ) -> None:
        """Fase de abejas Exploradoras: las fuentes que superan `limit`
        intentos sin mejorar (según la estrategia configurada) se
        abandonan y se reinician con una posición aleatoria uniforme."""
        to_reset = self.scout_fn(trial_counts, self.limit, rng) # Se seleccionan las fuentes que serán abandonas según la estrategia
        for idx in to_reset: # Por cada una 
            population[idx] = self.low + rng.random(self.n_vars) * (self.high - self.low) # Inicializamos la fuente en un punto aleatorio en el espacio de búsqueda
            fitness[idx] = float(self.problem.func(population[idx].reshape(1, -1))[0]) # Calculamos el fitness de la nueva fuente
            trial_counts[idx] = 0 # Reiniciamos el contador de estancamiento

    def run(self, rng: Generator | None = None, callback: CycleCallback | None = None) -> RunResult:
        """Ejecuta el ABC completo: cada ciclo aplica, en orden, la fase de
        empleadas, observadoras y exploradoras."""
        rng = rng if rng is not None else np.random.default_rng(self.config.seed) # Motor de números aleatorios
        stopping = build_stopping_criterion(self.config, self.bounds_arr) # Creación de los criterios de paro según las configuraciones y atributos

        population = self._random_population(self.colony_size, rng) # Generación aleatoria de las fuentes de alimento en el espacio de búsqueda
        fitness = self._evaluate(population) # Cálculo del fitness de las fuentes de alimento
        trial_counts = np.zeros(self.colony_size, dtype=np.int64) # Inicialización en 0's de los contadores de estancamiento

        # Estadística
        best_idx = int(np.argmin(fitness)) # Mejor fitness (Menor valor)
        best_x = population[best_idx].copy() # Mejor X (X para el menor Y alcanzada)
        best_fitness = float(fitness[best_idx]) # Mejor fitness alcanzado (Menor vaqlor)

        # Listas de convergencia y diversidad del problema
        convergence_best: list[float] = [] 
        diversity_history: list[float] = []

        cycle = 0 # Contador de iteraciones
        stopped_reason = "max_cycles" # Razón de paro (por default será número de ciclos; será modificada si otro criterio de paro es el que finalmente actua)

        while True: # Ciclo Principal
            current_best_idx = int(np.argmin(fitness)) # Mejor índice actual 
            if fitness[current_best_idx] < best_fitness: # Actualización del fitness y el mejor X dado el mejor índice (Menor valor)
                best_fitness = float(fitness[current_best_idx])
                best_x = population[current_best_idx].copy()

            convergence_best.append(best_fitness) # Índice de convergencia
            diversity_history.append(population_diversity_euclidean(population, self.bounds_arr)) # Historia de la evolución de la diversidad

            if callback is not None: # Función auxiliar si hay más información o procesos relevantes a medio ciclo. Lo usaremos para capturar 3 momentos de la población en una run
                callback(cycle, population, fitness)

            if stopping.should_stop(cycle, convergence_best, population): # Evaluación de los criterios de paro
                if cycle >= self.config.max_cycles: # Paro por máximo de ciclos (Paro Fuerte)
                    stopped_reason = "max_cycles"
                elif self.config.diversity_epsilon > 0 and diversity_history[-1] < self.config.diversity_epsilon: # Paro por colapso de diversidad (Paro extra)
                    stopped_reason = "diversity_collapse"
                else:
                    stopped_reason = "stagnation" # Paro débil
                break

            # Fases principales del ABD
            self._employed_phase(population, fitness, trial_counts, rng) # Fase de las abejas empleadas (Evaluación y búsqueda de mejores fuentes)
            self._onlooker_phase(population, fitness, trial_counts, rng) # Fase de las abejas observadoras (Selección de las mejores fuentes y búsqueda de mejores fuentes)
            self._scout_phase(population, fitness, trial_counts, rng) # Fase de las abejas exploradoras (Abdanonar fuentes estancadas y aleatorizar nuevas)
            cycle += 1
        #Retorno de resultados y estadística
        return RunResult(
            best_x=best_x,
            best_fitness=best_fitness,
            cycles_run=cycle,
            convergence_best=convergence_best,
            diversity_history=diversity_history,
            stopped_reason=stopped_reason,
        )


"""
 9. Utilidad de éxito
____________________________________________________________
"""

def is_success(problem: ProblemSpec, best_fitness: float, tolerance: float) -> bool | None:
    """Determina si una corrida fue "exitosa" comparando contra el óptimo
    analítico conocido. None si el problema no tiene óptimo conocido."""
    if problem.known_optimum is None:
        return None
    return abs(best_fitness - problem.known_optimum) < tolerance
