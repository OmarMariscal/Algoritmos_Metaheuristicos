from __future__ import annotations #Para evitar errores en importaciones

"""
Autor: Mariscal Rodríguez Omar Jesús
Materia: Algoritmos Metaheurísticos
Profesor: Paredes López Ángel Ignasio
Actividad 2 - Algoritmos Metaheurísticos

Universidad de Guadalajara
Centro Universitario de Ciencias Exactas e Ingenierías

------------------------------------------------------------------------

ga_binario.py
Este archivo es el motorr del Algoritmo Genético Binario para minimización de funciones
con dominio restringido tipo caja, dado que son el tipo de problemas que se enlistaron en la actividad.
A continuación, se detallan los componentes de la arquitectura:

Arquitectura:
    - GAConfig / ProblemSpec: dataclasses o contenedores de datos que centralizan toda la parametrización.
    - Encoding: codificación/decodificación binaria (N bits por variable | Configurable).
    - Operadores (selección, cruce, mutación): implementados como Protocols +
      registries, para poder intercambiarlos por nombre y evitar el antipatrón, if-else if anidados
    - StoppingCriterion: criterios de paro combinables (fuerte + débil).
    - PROBLEMS: registro de las funciones objetivo de la actividad, cada una
      con su dominio y (si se conoce) su óptimo global analítico para determinar si el algoritmo pudo llegar al mínimo esperado o no.

Este módulo está pensado para ser configurable y cambios no requieran tocar tanto código, sino que baste con:
    a) registrar una nueva función en el registry correspondiente, o
    b) cambiar un valor en GAConfig (Por ejemplo: selection="tournament").
"""

"""
Importaciones
"""

from dataclasses import dataclass, field #Dataclass como atajo para los contenedores de datos
from typing import Callable, Protocol, Sequence # Objetos necesarios para la arquitectura de Patrón de Estrategia

import numpy as np #Numpy para cálculos en matrices optimizados
from numpy.random import Generator #Motor de Números Aleatorios para el control de Semillas.

"""
1. Especificación del problema (función objetivo + dominio)
____________________________________________________________
"""

@dataclass(frozen=True)
class ProblemSpec:
    """
    Contenedor de Datos
    Describe un problema de minimización con dominio tipo caja.
    Se utiliza el frozen para no permitir su modificación
        -Las especificaciones de un problema no cambian a lo largo de la ejecución
    
    Attributos:
        name: Nombre identificador del problema (para automatizar la exportación de gráficas).
        func: Función objetivo. Recibe un array de forma (n_pop, n_vars) donde n_pop es el número de individuos y n_vars el número de variables y
            devuelve un array de forma (n_pop,) con el valor f(x) por individuo.
            (Vectorizada sobre la población para eficiencia.)
        bounds: Límites [(min, max), ...] por variable (dominio tipo caja).
                Es una lista ya que se permite que se tenga un limite por variable
        known_optimum: Valor óptimo global analítico, si se conoce (para
            calcular una tasa de éxito si se llego a la solución esperada). None si no aplica.
        n_vars: Número de variables (dimensión), inferido de `bounds`.
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    bounds: Sequence[tuple[float, float]]
    known_optimum: float | None = None

    @property
    def n_vars(self) -> int:
        return len(self.bounds)


"""
Problemas pedidos en la actividad
____________________________________________________________

Todas reciben un array de x en lugar de una variable simple para permitir funciones multivariable como la ackley
De esta manera, la primera fila corresponde a la primera variable, la segunda fila a la segunda variable (si aplica) y así sucesivamente según el problema
    Gracias a esto, estandarizamos el paso de variables y podemos trabajar más adelante sabiendo que esperamos en específico un tipo de dato y no múltiples
"""

def _f1(x: np.ndarray) -> np.ndarray:
    """f1(x) = x^3 + 4x^2 - 4x + 1, x en [-5, 3]."""
    xv = x[:, 0]
    return xv**3 + 4 * xv**2 - 4 * xv + 1


def _f2(x: np.ndarray) -> np.ndarray:
    """f2(x) = x^4 + 5x^3 + 4x^2 - 4x + 1, x en [-5, 3]."""
    xv = x[:, 0]
    return xv**4 + 5 * xv**3 + 4 * xv**2 - 4 * xv + 1


def _f3_ackley(x: np.ndarray) -> np.ndarray:
    """f3(x, y): función tipo Ackley, x, y en [-5, 5]."""
    xv, yv = x[:, 0], x[:, 1] #Extracción de las variables
    term1 = -20.0 * np.exp(-0.2 * np.sqrt((xv**2 + yv**2) / 2.0)) #Primer término de la función
    term2 = -np.exp((np.cos(2 * np.pi * xv) + np.cos(2 * np.pi * yv)) / 2.0) #Segundo término de la función
    return np.e + term1 + term2 + 20.0 #Forma final de la función ackley

def _f4_rastberry (x: np.ndarray) -> np.ndarray:
    xv, yv = x[:,0], x[:,1]
    return 20 + xv**2 + yv**2 - 10*(np.cos(2 * np.pi * xv) + np.cos(2 * np.pi * yv))
""""
Registro de problemas de la actividad.
Los Nuevos problemas se agregan aquí sin modificar el resto del código.

Usamos el contenedor definido antes para el especificador de problema
Óptimos globales analíticos (obtenidos por cálculo diferencial, verificando
también los extremos del dominio ya que son restricciones tipo caja):

  f1: f'(x) = 3x^2 + 8x - 4 = 0 -> x = 0.4305 (mínimo local, f=0.0991)
      pero el mínimo GLOBAL en [-5, 3] está en la frontera x = -5, f(-5) = -4
  f2: f'(x) = 4x^3 + 15x^2 + 8x - 4 = 0 -> tres raíces reales; el mínimo
      global en [-5, 3] es el mínimo local interior x = -2.9603, f = -5.0196
      (los extremos del dominio dan valores mucho mayores: 121 y 241)
  f3 (Ackley): óptimo global conocido en (0, 0), f(0,0) = 0
"""

PROBLEMS: dict[str, ProblemSpec] = {
    "f1": ProblemSpec(name="f1", func=_f1, bounds=[(-5.0, 3.0)], known_optimum=-4.0),
    "f2": ProblemSpec(name="f2", func=_f2, bounds=[(-5.0, 3.0)], known_optimum=-5.019646349962712),
    "f3_ackley": ProblemSpec(
        name="f3_ackley", func=_f3_ackley, bounds=[(-5.0, 5.0), (-5.0, 5.0)], known_optimum=0.0
    ),
    "f4_rastringin": ProblemSpec(name= 'f4_rastringin', func=_f4_rastberry, bounds=[(-5.12, 5.12), [-5.12, 5.12]])
}


"""
2. Configuración del algoritmo
____________________________________________________________
"""

@dataclass
class GAConfig:
    """
    Hiperparámetros y opciones del Algoritmo. 
    Un único objeto que viaja por
    todo el motor, para facilitar cambios sin tocar firmas de función.

    Attributos:
        bits_per_var: Bits usados para codificar cada variable (Pensado para 4 u 8).
        population_size: Tamaño de la población.
        max_generations: Criterio de paro "fuerte" en número de generaciones o iteraciones..
        crossover_prob: Probabilidad de cruce.
        mutation_prob: Probabilidad de mutación por bit.
        elitism: Número de mejores individuos que pasan sin alterar a la
            siguiente generación (0 es sin elitismo).
        selection: Nombre del operador de selección registrado en SELECTION_OPERATORS (como ruleta o torneo)
        crossover: Nombre del operador de cruce registrado en CROSSOVER_OPERATORS (Como cruce en un punto o cruce uniforme).
        mutation: Nombre del operador de mutación registrado en MUTATION_OPERATORS (Dinámica para decidir y aplicar la mutación en los hijos).
        tournament_k: Tamaño del torneo, usado solo si selection="tournament".
        stagnation_generations: Generaciones sin mejora significativa para
            disparar el criterio de paro "débil" (0 = desactivado), esto está hecho para evitar estancamiento.
        stagnation_epsilon: Mejora mínima considerada significativa para
            resetear el contador de estancamiento.
        success_tolerance: Tolerancia |f_encontrado - f_óptimo| para contar
            una corrida como "exitosa" (requiere known_optimum en ProblemSpec).
        seed: Semilla para reproducibilidad (uso aquí numpy.random.Generator).

    El Config tiene algunos valores por defecto vistos en clase
            
    """


    bits_per_var: int = 8
    population_size: int = 50
    max_generations: int = 200
    crossover_prob: float = 0.85
    mutation_prob: float = 0.03
    elitism: int = 1
    selection: str = "roulette"
    crossover: str = "one_point"
    mutation: str = "bit_flip"
    tournament_k: int = 3
    stagnation_generations: int = 20
    stagnation_epsilon: float = 1e-6
    success_tolerance: float = 1e-3
    seed: int | None = None
    porcentual_elitism: bool = False


"""
# 3. Codificación / decodificación binaria
____________________________________________________________
"""

class Encoding:
    """Codificación binaria de manera vectorial por optimización.

    Cada variable se codifica con `bits_per_var` bits. Un cromosoma de un
    individuo con n_vars variables tiene longitud n_vars * bits_per_var.
    """

    def __init__(self, bounds: Sequence[tuple[float, float]], bits_per_var: int) -> None:
        self.bounds = list(bounds)
        self.bits_per_var = bits_per_var
        self.n_vars = len(bounds)
        self.chromosome_length = self.n_vars * bits_per_var
        self._max_int = 2**bits_per_var - 1

    def decode(self, population: np.ndarray) -> np.ndarray:
        """Decodifica una población binaria (n_pop, chromosome_length) a
        valores reales (n_pop, n_vars) dentro de los límites de cada variable.
        """

        n_pop = population.shape[0] # Obtenemos el número de individuos
        decoded = np.empty((n_pop, self.n_vars), dtype=np.float64) # Creamos un vector de tamaño (individuos x n_variables)
        weights = (2 ** np.arange(self.bits_per_var - 1, -1, -1)).astype(np.float64) # Creamos un array con los pesos de las potencias de 2 de manera decreciente 
        # Ejemplo, si la población si tenemos 4 bits por variable, weights será igual a [8, 4, 2, 1] es decir [2^3, 2^2, 2^1, 2^0]
        # Esto está hecho para hacer la decodificación más rápida con un producto punto.
        # Imaginando que el vector de población es [1 0 1 1] se hará el producto punto con [8, 4, 2, 1]
        # Resultando en 11
        # es más eficiente así con Numpy que calculando individualmente las potencias una a una.

        # Ciclo que se repetirá la cantidad de variables que haya (inferido desde los límites)
        for i, (low, high) in enumerate(self.bounds):
            start = i * self.bits_per_var # Punto inicial de la población de la variable n
            end = start + self.bits_per_var # Punto final de la población de la variable n
            segment = population[:, start:end] # Extracción de los individuos que pertenecen a la variable n
            int_values = segment @ weights  # Producto punto que resulta en un arreglo (n_pop,) con la variable
            decoded[:, i] = low + (int_values / self._max_int) * (high - low) # Usamos la fórmula para transformar la decodificación a su equivalente en el rango establecido de la variable (es como una normalización)

        return decoded #Retornamos el vector de los valores decodificados

    """
    Generar una población aleatoria inicial
    """
    def random_population(self, size: int, rng: Generator) -> np.ndarray:
        """Genera una población binaria aleatoria (size, chromosome_length)."""
        return rng.integers(0, 2, size=(size, self.chromosome_length), dtype=np.int8)


"""
# 4. Operadores genéticos (Protocol + registry. Intercambiables)
____________________________________________________________
"""

"""
Protocolos o Contratos con una estructura definida para la arquitectura de Patrón de Estrategia
"""

class SelectionOperator(Protocol):
    def __call__( #__call__ permite que se llame a una función como si fuera un método, ahorra algo de código
        self, population: np.ndarray, fitness: np.ndarray, n_select: int, rng: Generator, config: GAConfig
    ) -> np.ndarray:
        """Devuelve índices (n_select,) de individuos seleccionados como padres."""
        ...


class CrossoverOperator(Protocol):
    def __call__(
        self, parent_a: np.ndarray, parent_b: np.ndarray, rng: Generator, config: GAConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cruza dos cromosomas y devuelve dos hijos."""
        ...


class MutationOperator(Protocol):
    def __call__(self, chromosome: np.ndarray, rng: Generator, config: GAConfig) -> np.ndarray:
        """Aplica mutación in-place-friendly (devuelve el cromosoma mutado)."""
        ...


def _fitness_from_minimization(raw_fitness: np.ndarray) -> np.ndarray:
    """Convierte valores de f(x) (a minimizar) en "aptitud" positiva para
    selección proporcional (ruleta): mayor aptitud = mejor individuo.
    Usa un desplazamiento para evitar aptitudes negativas o nulas.

    Se usa como primer paso para convertir el fitness en porcentajes
    """
    worst = raw_fitness.max() #En una minimización, el peor individuo es el del fitness más alto
    # +1e-9 evita que el peor individuo tenga aptitud exactamente 0
    return (worst - raw_fitness) + 1e-9


def select_roulette(
    population: np.ndarray, fitness: np.ndarray, n_select: int, rng: Generator, config: GAConfig
) -> np.ndarray:
    """Selección por ruleta (proporcional a la aptitud), con `fitness` como
    valores de f(x) a MINIMIZAR (se transforman internamente a aptitud)."""
    adjusted = _fitness_from_minimization(fitness) # Ajustamos el fitness a la minimización
    probs = adjusted / adjusted.sum() # Lo convertimos en porcentajes
    return rng.choice(len(population), size=n_select, replace=True, p=probs) #Aleatoriamente elegimos en base a los porcentajes


def select_tournament(
    population: np.ndarray, fitness: np.ndarray, n_select: int, rng: Generator, config: GAConfig
) -> np.ndarray:
    """Selección por torneo de tamaño k (config.tournament_k). Gana el
    individuo con menor f(x) (problema de minimización)."""
    k = max(2, config.tournament_k) #Número de participantes (Mínimo 2)
    n_pop = len(population) # Obtenemos el número de individuos
    selected = np.empty(n_select, dtype=np.int64) # Prepramos la matriz vacía donde irán los ganadores
    for i in range(n_select): # Bucle que seacrá n_select ganadores
        contenders = rng.integers(0, n_pop, size=k) # Los contendientes se eligen al azar 
        winner = contenders[np.argmin(fitness[contenders])] # Tomamos el ganador (El mínimo)
        selected[i] = winner # Agregamos el ganador a la lista
    return selected

def stochastic_tournament(population: np.ndarray, fitness: np.ndarray, n_select: int, rng: Generator, config: GAConfig):
    k = max(2, config.tournament_k)
    n_pop = len(population)
    selected = np.empty(n_select, dtype=np.int64)
    for i in range(n_select):
        contenders = rng.integers(0, n_pop, size=k)

        # Selección por Porcentaje (Estilo Ruleta)
        adjusted = _fitness_from_minimization(fitness[contenders])
        probs = adjusted / adjusted.sum()

        winner = rng.choice(contenders, p=probs)

        selected[i] = winner
    return selected

SELECTION_OPERATORS: dict[str, SelectionOperator] = {
    "roulette": select_roulette,
    "tournament": select_tournament,
    "stochastic_tournament": stochastic_tournament,
}


def crossover_one_point(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: Generator, config: GAConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce de un punto. Con probabilidad (1 - Pc) los hijos son copias
    exactas de los padres."""

    # El cruce puede darse o no según la probabilidad asingada
    if rng.random() >= config.crossover_prob:
        return parent_a.copy(), parent_b.copy()

    length = len(parent_a) #Obtenemos el número de bits por cada padre
    if length < 2: # Si es 1, no hacemos cruza
        return parent_a.copy(), parent_b.copy()

    point = rng.integers(1, length) # El punto se elige aleatoriamente en el rango 
    child_a = np.concatenate([parent_a[:point], parent_b[point:]]) #El hijo A hereda la primera sección del padre A y la segunda del padre B
    child_b = np.concatenate([parent_b[:point], parent_a[point:]]) #El hijo B hereda la primera sección del padre B y la segunda del padre A
    return child_a, child_b


def crossover_uniform(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: Generator, config: GAConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce uniforme: cada bit se hereda de un padre u otro con prob. 0.5.
    Incluido como alternativa lista para usarse ante cambios solicitados."""

    #Probabilidad de cruce
    if rng.random() >= config.crossover_prob:
        return parent_a.copy(), parent_b.copy()

    mask = rng.integers(0, 2, size=len(parent_a)).astype(bool) # Generamos aletoriamente un arreglo de boolenaos de longitud igual, los true donde habrá cruce
    child_a = np.where(mask, parent_a, parent_b) # Para el hijo A, losTrue toman el valor de A en esa posición y los que están en False tomaran el valor de B en esa posición
    child_b = np.where(mask, parent_b, parent_a) # Para el hijo B, los True toman el valor de B en esa posición y los que están en False tomaran el valor de A.
    return child_a, child_b


CROSSOVER_OPERATORS: dict[str, CrossoverOperator] = {
    "one_point": crossover_one_point,
    "uniform": crossover_uniform,
}


def mutate_bit_flip(chromosome: np.ndarray, rng: Generator, config: GAConfig) -> np.ndarray:
    """Mutación estándar: cada bit se invierte independientemente con
    probabilidad Pm."""
    flips = rng.random(len(chromosome)) < config.mutation_prob # Probabilidad de que se dé la mutación (calculada para cada bit)
    mutated = chromosome.copy() # Guardamos el registro no alterando el cromosoma de origen
    mutated[flips] = 1 - mutated[flips] # Los que se les aplicará el bit, se les invierte el valor (con 1-valorBit)
    return mutated


MUTATION_OPERATORS: dict[str, MutationOperator] = {
    "bit_flip": mutate_bit_flip,
}



"""
# 5. Criterios de paro (combinables: fuerte + débil)
____________________________________________________________
"""

"""
Contrato con el método should_stop para los métodos de paro
"""
class StoppingCriterion(Protocol):
    def should_stop(self, generation: int, best_history: list[float]) -> bool:
        ...


@dataclass
class MaxGenerationsStopping:
    """Criterio de paro FUERTE: tope duro de generaciones."""

    max_generations: int

    def should_stop(self, generation: int, best_history: list[float]) -> bool:
        return generation >= self.max_generations


@dataclass
class StagnationStopping:
    """Criterio de paro DÉBIL: se detiene si el mejor fitness no mejora
    más que `epsilon` durante `patience` generaciones consecutivas.
    Si `patience <= 0`, el criterio queda desactivado (nunca detiene).
    
    Está hecho para evitar estancamiento
    """

    patience: int
    epsilon: float

    def should_stop(self, generation: int, best_history: list[float]) -> bool:
        if self.patience <= 0 or len(best_history) <= self.patience:
            return False
        recent = best_history[-(self.patience + 1):]
        improvement = recent[0] - min(recent)  # minimización: mejora = baja el valor
        return improvement < self.epsilon


@dataclass
class CombinedStopping:
    """Combina varios criterios: se detiene si CUALQUIERA se cumple."""

    criteria: list[StoppingCriterion]

    def should_stop(self, generation: int, best_history: list[float]) -> bool:
        return any(c.should_stop(generation, best_history) for c in self.criteria) # Ciclo en una línea que se cumple si cualquier criterior se cumple


def build_stopping_criterion(config: GAConfig) -> CombinedStopping:
    """Arma el criterio de paro combinado (fuerte + débil) a partir de GAConfig."""
    return CombinedStopping(
        criteria=[
            MaxGenerationsStopping(max_generations=config.max_generations),
            StagnationStopping(
                patience=config.stagnation_generations, epsilon=config.stagnation_epsilon
            ),
        ]
    )

"""
# 6. Resultado de una corrida + callbacks
____________________________________________________________
"""

@dataclass
class RunResult:
    """
    Contenedor de datos para los resultados de una ejecución del algoritmo
    """

    best_x: np.ndarray # Arreglo con el mejor resultado de cada generación
    best_fitness: float # El mejor fitness global
    generations_run: int # Número de generaciones alcanzadas
    convergence_best: list[float] = field(default_factory=list)   # mejor fitness por generación
    convergence_mean: list[float] = field(default_factory=list)   # fitness medio por generación
    stopped_reason: str = "" # Logs de por qué paró el algoritmo


GenerationCallback = Callable[[int, np.ndarray, np.ndarray], None]


"""
7. Motor evolutivo 
Se ejecuta ignorando los procesos y configuraciones que se utilizan gracias a la arquitectura de Protocolo de Estrategia
____________________________________________________________
"""


class GeneticAlgorithm:
    """Motor del Algoritmo Genético Binario. No conoce los detalles de los
    operadores concretos: los obtiene de los registries según `config`.
    Esto permite intercambiar selección/cruce/mutación cambiando solo
    `GAConfig`, o agregando una entrada nueva a un registry.
    """

    def __init__(self, problem: ProblemSpec, config: GAConfig) -> None:
        self.problem = problem
        self.config = config
        self.encoding = Encoding(problem.bounds, config.bits_per_var)

        # Validar que las selecciones existen dentro de las listas
        try:
            self.selection_fn = SELECTION_OPERATORS[config.selection]
        except KeyError as e:
            raise ValueError(
                f"Operador de selección '{config.selection}' no registrado. "
                f"Disponibles: {list(SELECTION_OPERATORS)}"
            ) from e

        try:
            self.crossover_fn = CROSSOVER_OPERATORS[config.crossover]
        except KeyError as e:
            raise ValueError(
                f"Operador de cruce '{config.crossover}' no registrado. "
                f"Disponibles: {list(CROSSOVER_OPERATORS)}"
            ) from e

        try:
            self.mutation_fn = MUTATION_OPERATORS[config.mutation]
        except KeyError as e:
            raise ValueError(
                f"Operador de mutación '{config.mutation}' no registrado. "
                f"Disponibles: {list(MUTATION_OPERATORS)}"
            ) from e

    #Función para evaluar la población
    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        decoded = self.encoding.decode(population)
        return self.problem.func(decoded)

    #Binario a Gray
    def binaryToGray(self, binary_chromosome: np.ndarray) -> np.ndarray:
        """Transforma un único cromosoma de bits binarios a código Gray."""
        shifted = np.roll(binary_chromosome, shift=1)
        shifted[0] = 0  # El bit más significativo no cambia
        return binary_chromosome ^ shifted

    def graytoBinary(self, gray_chromosome: np.ndarray) -> np.ndarray:
        """Transforma un único cromosoma en código Gray de vuelta a bits binarios."""
        binary_chromosome = np.zeros_like(gray_chromosome)

        binary_chromosome[0] = gray_chromosome[0]        
        for i in range(1, len(gray_chromosome)):
            binary_chromosome[i] = binary_chromosome[i - 1] ^ gray_chromosome[i]
            
        return binary_chromosome

    def run(
        self,
        rng: Generator | None = None,
        callback: GenerationCallback | None = None,
    ) -> RunResult:
        """Ejecuta el AGB completo y devuelve el resultado con historial de
        convergencia (mejor y media por generación).

        Parámetros:
            rng: Generador de números aleatorios. Si es None, se crea uno
                nuevo a partir de config.seed.
            callback: Función opcional invocada al final de cada generación
                con (generation, population, fitness) — útil para logging o
                análisis adicional sin modificar el motor.
        """
        rng = rng if rng is not None else np.random.default_rng(self.config.seed) # Establecer el motor de random
        stopping = build_stopping_criterion(self.config) # Crear el criterio de paro combinado según el config

        population = self.encoding.random_population(self.config.population_size, rng) # Generamos la población aleatoria
        fitness = self._evaluate(population) # Primera evaluación del Fitness de la población aleatoria

        # Listas donde iremos guardando datos útiles para las estadísticas
        convergence_best: list[float] = []
        convergence_mean: list[float] = []

        generation = 0 # Contador de generaciones
        stopped_reason = "max_generations" # Razón de Paro, Inicialmente establecida en máximo de generaciones o iteraciones pero se modificará si llega a cambiar

        while True: # Bucle Principal
            #Agregamos las estadísticas a las listas
            convergence_best.append(float(fitness.min())) 
            convergence_mean.append(float(fitness.mean()))

            if callback is not None:
                callback(generation, population, fitness) # Función de Logs opcional

            # Evaluación de los criterios de Paro
            if stopping.should_stop(generation, convergence_best):
                if generation >= self.config.max_generations:
                    stopped_reason = "max_generations"
                else:
                    stopped_reason = "stagnation"
                break

            # Obtenemos la siguiente generación una vez pasa por el proceso de selección, cruce y mutación (se detalla a continuación)
            population, fitness = self._next_generation(population, fitness, rng)
            generation += 1 #Aumentamos el conteo de generaciones

        best_idx = int(np.argmin(fitness)) # Encontramos el índice del mejor resultado (el menor)
        best_x = self.encoding.decode(population[best_idx : best_idx + 1])[0] # Obtenemos el resultado del mejor resultado después de que el algoritmo parara ya decodificada

        # Construimos el contenedor de datos con los resultados de una vez el algoritmo
        return RunResult(
            best_x=best_x,
            best_fitness=float(fitness[best_idx]),
            generations_run=generation,
            convergence_best=convergence_best,
            convergence_mean=convergence_mean,
            stopped_reason=stopped_reason,
        )

    # Función que, en base al fitness y a la población, realiza crucers y mutaciones para calcular la siguiente generación
    def _next_generation(
        self, population: np.ndarray, fitness: np.ndarray, rng: Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n_pop = len(population)
        config = self.config

        # Elitismo(Si aplica): los mejores individuos pasan sin alterar
        if(config.porcentual_elitism):
            elite_count = 2 if n_pop < 100 else n_pop // 2
        else:
            elite_count = max(0, min(config.elitism, n_pop)) # Asginación segura en 0 para evitar negativos

        elite_idx = np.argsort(fitness)[:elite_count] # Encontramos el o los mejores individuos de la población anterior
        new_population = [population[i].copy() for i in elite_idx] # Los pasamos directamente a la nueva generación

        # Bucle principal hasta completar la población
        while len(new_population) < n_pop:
            parents_idx = self.selection_fn(population, fitness, 2, rng, config) # Pasamos por el proceso de selección de los padres y obtenemos sus índices
            parent_a_binary, parent_b_binary = population[parents_idx[0]], population[parents_idx[1]] # Extraemos los padres de la población

            parent_a_gray = self.binaryToGray(parent_a_binary)
            parent_b_gray = self.binaryToGray(parent_b_binary)

            child_a_gray, child_b_gray = self.crossover_fn(parent_a_gray, parent_b_gray, rng, config) # Cruzamos los padres
            child_a_gray = self.mutation_fn(child_a_gray, rng, config) # Cada hijo pasa por la mutación
            child_b_gray = self.mutation_fn(child_b_gray, rng, config)

            child_a_binary = self.graytoBinary(child_a_gray)
            child_b_binary = self.graytoBinary(child_b_gray)

            new_population.append(child_a_binary) # Agregamos primero un hijo y luego el otro para evitar que la población crezca en número (Puede modificarse para hacer aleatorio en caso de que no alcancen los dos hijos, pase el A o el B)
            if len(new_population) < n_pop:
                new_population.append(child_b_binary)

        new_population_arr = np.array(new_population, dtype=np.int8) # Para mantener historial, creamos un arreglo copiando el de new_population
        new_fitness = self._evaluate(new_population_arr) # Lo pasamos por la evaluación donde tendremos un arreglo con los fitness de esta nueva población
        return new_population_arr, new_fitness # Retornaqmos la población y los fitness que serán usados por _run para las estadísticas


"""
# 8. Utilidad de éxito (usa el óptimo analítico si está disponible)
____________________________________________________________
"""

def is_success(problem: ProblemSpec, best_fitness: float, tolerance: float) -> bool | None:
    """Determina si una corrida fue "exitosa" comparando contra el óptimo
    analítico conocido. Devuelve None si el problema no tiene óptimo
    conocido (en ese caso, la tasa de éxito no puede calcularse así)."""
    if problem.known_optimum is None:
        return None
    return abs(best_fitness - problem.known_optimum) < tolerance