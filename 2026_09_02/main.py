"""
Autor: Mariscal Rodríguez Omar Jesús
Materia: Algoritmos Metaheurísticos
Profesor: Paredes López Ángel Ignasio
Actividad 2 - Algoritmos Metaheurísticos

Universidad de Guadalajara
Centro Universitario de Ciencias Exactas e Ingenierías

------------------------------------------------------------------------

main.py

Orquesta el experimento completo de la Actividad 2:

    - Para cada función objetivo (para el caso de esta tarea fueron tituladas como: f1, f2, f3_ackley):
        - Para cada combinación de (bits_per_var, mutation_prob) en
          {4, 8} x {1%, 3%, 5%}  (6 combinaciones):
            - Ejecuta el Algoritmo Evolutivo Binario N_RUNS veces con semillas distintas.
            - Registra: mejor fitness, curva de convergencia (mejor y media).
    - Calcula estadísticas (mejor, peor, media, desviación estándar, tasa
      de éxito) por función y combinación.
    - Genera:
        - Tabla resumen (impresa en consola + exportada a CSV para ahorrar trabajo).
        - Gráficas de convergencia PROMEDIO (media de las N_RUNS corridas,
          no solo la mejor corrida) por función y combinación.
            -Como nota, para simplificar las gráficas pedidas en la actividad donde se pide una gráfica por corrida 
            del algoritmo (combinación {N_BITS, PM}) se genera solo una con las 6 combinaciones posibles en cada gráfica de cada función

Para modificaciones y aprovehcar la arquitectura puede hacerse de la siguiente manera:
    - Cambiar operador de selección: usar --selection tournament (o editar
      DEFAULT_CONFIG_OVERRIDES) -> internamente solo cambia GAConfig.selection,
      el motor (ga_binario.GeneticAlgorithm) no requiere ningún otro cambio.
        -El script está configurado para funcionar en consola con args con lo ya establecido.
    - Cambiar la función de aptitud: agregar/editar una entrada en
      ga_binario.PROBLEMS, o pasar otra función compatible con la firma
      Callable[[np.ndarray], np.ndarray].

Ejemplo de Uso:
    python main.py                       # corrida completa (20 runs x 6 combos x 3 funciones)
    python main.py --runs 5              # corrida rápida de prueba
    python main.py --selection tournament --tournament-k 3
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # backend sin GUI, apto para entorno de script/servidor
import matplotlib.pyplot as plt
import numpy as np

from ga_binario import (
    GAConfig,
    GeneticAlgorithm,
    PROBLEMS,
    ProblemSpec,
    RunResult,
    is_success,
)

"""
# Configuración del experimento
____________________________________________________________
Combinaciones de Configuraciones pedidas en la actividad
"""

BITS_LEVELS: list[int] = [4, 8]
MUTATION_RATES: list[float] = [0.01, 0.03, 0.05]
N_RUNS_DEFAULT = 20
OUTPUT_DIR = "resultados"


@dataclass
class CombinationResult:
    """Resultado agregado de una combinación (bits, Pm) para una función:
    N_RUNS corridas completas.
    
    Es decir, este almacena el resultado de las 20 ejecuciones del algoritmo bajo una misma configuración
    """

    problem_name: str
    bits: int
    mutation_rate: float
    runs: list[RunResult] = dataclasses.field(default_factory=list) # Lista de Resultados de las Corridas

    @property
    def best_fitness_values(self) -> np.ndarray:
        return np.array([r.best_fitness for r in self.runs])

    def mean_convergence_curve(self, kind: str = "mean") -> np.ndarray:
        """Promedia, generación a generación, la curva de convergencia de
        todas las corridas (kind: 'mean' usa el fitness medio poblacional
        por generación de cada corrida; 'best' usa el mejor por generación).
        Las corridas pueden tener distinta longitud (paro débil), así que
        se rellena con el último valor (curva ya convergida) antes de promediar.
        """
        curves = [r.convergence_mean if kind == "mean" else r.convergence_best for r in self.runs]
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        return padded.mean(axis=0)

"""
# Ejecución del experimento
____________________________________________________________
"""

def run_experiment(
    problems: dict[str, ProblemSpec],
    n_runs: int,
    base_seed: int,
    config_overrides: dict,
) -> dict[tuple[str, int, float], CombinationResult]:
    """Corre las 6 combinaciones x N_RUNS corridas para cada función.
    Devuelve un dict indexado por (problem_name, bits, mutation_rate).
    """
    results: dict[tuple[str, int, float], CombinationResult] = {} # Diccionario que almacenará los resultados de cada corrida
    rng_seeder = np.random.default_rng(base_seed) # Semilla Autogenerada

    total_combos = len(problems) * len(BITS_LEVELS) * len(MUTATION_RATES) # Calculo de cuantas veces diferentes se ejecutará el experimento
    combo_i = 0 # Contador de Combinación

    # Bucle Principal que recorre los problemas
    for problem_name, problem in problems.items():
        for bits in BITS_LEVELS: # Bucle Anidado para ejecutar por número de Bits
            for pm in MUTATION_RATES: # Bucle Anidado para ejecutar por porcentaje de mutación
                combo_i += 1 # Contador de combinación
                combo_result = CombinationResult(problem_name=problem_name, bits=bits, mutation_rate=pm) # Creamos un contenedor de datos que albergue las características del problema 
                print(f"[{combo_i}/{total_combos}] {problem_name} | bits={bits} | Pm={pm:.0%} "
                      f"-> {n_runs} corridas...", end="", flush=True) # Impresión a manera de tabla en terminal

                t0 = time.time() # Calculo de cuánto tiempo toma cada combinación (para comprobar que es eficiente)
                for run_idx in range(n_runs): # Bucle Anidado para cada una de las 20 corridas del algoritmo
                    run_seed = int(rng_seeder.integers(0, 2**31 - 1)) # Semilla Aleatoria en límite en el valor decimal de 31 bits dado que np trabaja en una base en c con un entero con signo, 
                    # Creamos una configuración en base a las características del problema
                    config = GAConfig( 
                        bits_per_var=bits,
                        mutation_prob=pm,
                        seed=run_seed,
                        **config_overrides,
                    )
                    ga = GeneticAlgorithm(problem, config) # Creamos el motor del algoritmo con las configuraciones
                    result = ga.run(rng=np.random.default_rng(run_seed)) # Ejecutamos el algoritmo 
                    combo_result.runs.append(result) # Agregamos los resultados

                elapsed = time.time() - t0 # Una vez acaban las 20 ejecuciones, paramos el tiempo
                print(f" listo ({elapsed:.1f}s)") # Impresiones en pantalla
                results[(problem_name, bits, pm)] = combo_result # Agregamos los resultados finales del combo

    return results # Retornamos los resultados


"""
Estadísticas
____________________________________________________________
"""

"""
Contenedor de Datos de las Estadísticas
"""
@dataclass
class StatsRow:
    problem_name: str
    bits: int
    mutation_rate: float
    best: float
    worst: float
    mean: float
    std: float
    success_rate: float | None  # None si el problema no tiene óptimo conocido


"""
En base a los resultados, calculamos las estadísticas que se piden
"""
def compute_stats(
    results: dict[tuple[str, int, float], CombinationResult],
    problems: dict[str, ProblemSpec],
    tolerance: float,
) -> list[StatsRow]:
    rows: list[StatsRow] = []
    #Bucle que descompone los combos 
    for (problem_name, bits, pm), combo in results.items():
        values = combo.best_fitness_values
        problem = problems[problem_name]

        successes = [is_success(problem, v, tolerance) for v in values]
        if problem.known_optimum is None or any(s is None for s in successes):
            success_rate = None
        else:
            success_rate = float(np.mean(successes))

        rows.append(
            StatsRow(
                problem_name=problem_name,
                bits=bits,
                mutation_rate=pm,
                best=float(values.min()),
                worst=float(values.max()),
                mean=float(values.mean()),
                std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                success_rate=success_rate,
            )
        )

    # Orden estable: por función, luego bits, luego tasa de mutación
    rows.sort(key=lambda r: (r.problem_name, r.bits, r.mutation_rate))
    return rows


#Imprimir la tabla de resultados en términal de forma legible
def print_stats_table(rows: list[StatsRow]) -> None:
    header = f"{'Función':<10} {'Bits':>5} {'Pm':>6} {'Mejor':>12} {'Peor':>12} {'Media':>12} {'Desv.Std':>12} {'% Éxito':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        success_str = f"{r.success_rate:.0%}" if r.success_rate is not None else "N/A"
        print(
            f"{r.problem_name:<10} {r.bits:>5} {r.mutation_rate:>5.0%} "
            f"{r.best:>12.6f} {r.worst:>12.6f} {r.mean:>12.6f} {r.std:>12.6f} {success_str:>9}"
        )

# Exportar a un csv para ponerla en el reporte de práctica
def export_stats_csv(rows: list[StatsRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["funcion", "bits", "Pm", "mejor", "peor", "media", "desv_std", "tasa_exito"])
        for r in rows:
            writer.writerow([
                r.problem_name,
                r.bits,
                r.mutation_rate,
                f"{r.best:.10f}",
                f"{r.worst:.10f}",
                f"{r.mean:.10f}",
                f"{r.std:.10f}",
                f"{r.success_rate:.4f}" if r.success_rate is not None else "N/A",
            ])
    print(f"\nTabla estadística exportada a: {path}")


"""
Gráficas de convergencia promedio
____________________________________________________________
"""

def plot_convergence(
    results: dict[tuple[str, int, float], CombinationResult],
    problem_name: str,
    output_path: str,
) -> None:
    """Una figura por función, con una curva por cada combinación
    (bits, Pm), mostrando la convergencia PROMEDIO (fitness medio
    poblacional promediado sobre las N_RUNS corridas) por generación."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for bits in BITS_LEVELS:
        for pm in MUTATION_RATES:
            combo = results[(problem_name, bits, pm)]
            curve = combo.mean_convergence_curve(kind="mean")
            ax.plot(curve, label=f"{bits} bits, Pm={pm:.0%}", linewidth=1.5)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness medio poblacional (promedio de las corridas)")
    ax.set_title(f"Curva de convergencia promedio — {problem_name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de convergencia guardada: {output_path}")


def plot_best_convergence(
    results: dict[tuple[str, int, float], CombinationResult],
    problem_name: str,
    output_path: str,
) -> None:
    """Complementaria: convergencia promedio del MEJOR fitness por
    generación (útil para ver qué tan rápido se acerca al óptimo)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for bits in BITS_LEVELS:
        for pm in MUTATION_RATES:
            combo = results[(problem_name, bits, pm)]
            curve = combo.mean_convergence_curve(kind="best")
            ax.plot(curve, label=f"{bits} bits, Pm={pm:.0%}", linewidth=1.5)

    problem = PROBLEMS[problem_name]
    if problem.known_optimum is not None:
        ax.axhline(problem.known_optimum, color="black", linestyle="--", linewidth=1, label="Óptimo global")

    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor fitness (promedio de las corridas)")
    ax.set_title(f"Convergencia del mejor fitness — {problem_name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de mejor-fitness guardada: {output_path}")


"""
CLI 

parse_args para que el algoritmo pueda funcionar como script configurable en terminal
____________________________________________________________
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actividad 2 - AGB: experimento completo")
    parser.add_argument("--runs", type=int, default=N_RUNS_DEFAULT, help="Corridas por combinación (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base para generar semillas de cada corrida")
    parser.add_argument("--population", type=int, default=50, help="Tamaño de población")
    parser.add_argument("--max-generations", type=int, default=200, help="Tope de generaciones (paro fuerte)")
    parser.add_argument("--crossover-prob", type=float, default=0.85, help="Probabilidad de cruce (Pc)")
    parser.add_argument("--elitism", type=int, default=1, help="Número de individuos elite")
    parser.add_argument("--selection", type=str, default="roulette", choices=["roulette", "tournament", "stochastic_tournament"],
                         help="Operador de selección")
    parser.add_argument("--crossover", type=str, default="one_point", choices=["one_point", "uniform"],
                         help="Operador de cruce")
    parser.add_argument("--tournament-k", type=int, default=3, help="Tamaño de torneo (si selection=tournament)")
    parser.add_argument("--stagnation-generations", type=int, default=20,
                         help="Paro débil: generaciones sin mejora (0 = desactivado)")
    parser.add_argument("--stagnation-epsilon", type=float, default=1e-6, help="Mejora mínima significativa")
    parser.add_argument("--success-tolerance", type=float, default=1e-3, help="Tolerancia para tasa de éxito")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Carpeta de salida")
    parser.add_argument("--porcentual-elitism", action="store_true", default=False, help="Actual el Elitismo Porcentual")

    return parser.parse_args()


# Función principal
def main() -> None:
    args = parse_args() # Obtenemos lo que se obtuvo de la terminal
    os.makedirs(args.output_dir, exist_ok=True) # Crear el directorio donde se almacenan las gráficas y el CSV

    config_overrides = dict( # Creamos un objeto de configuración en base a la entrada de terminal
        population_size=args.population,
        max_generations=args.max_generations,
        crossover_prob=args.crossover_prob,
        elitism=args.elitism,
        selection=args.selection,
        crossover=args.crossover,
        tournament_k=args.tournament_k,
        stagnation_generations=args.stagnation_generations,
        stagnation_epsilon=args.stagnation_epsilon,
        success_tolerance=args.success_tolerance,
        porcentual_elitism=args.porcentual_elitism 
    )

    # Inicio de las impresiones
    print("=" * 70)
    print("Actividad 2 - Algoritmo Genético Binario")
    print(f"Corridas por combinación: {args.runs} | Selección: {args.selection} | "
          f"Cruce: {args.crossover} | Elitismo: {args.elitism}")
    print("=" * 70)

    # Ejecutamos el experimento las con todos los problemas y las configuraciones
    results = run_experiment(
        problems=PROBLEMS,
        n_runs=args.runs,
        base_seed=args.seed,
        config_overrides=config_overrides,
    )

    stats_rows = compute_stats(results, PROBLEMS, tolerance=args.success_tolerance) # Obtenemos las estadísticas
    print_stats_table(stats_rows) # Imprimps resultados
    export_stats_csv(stats_rows, os.path.join(args.output_dir, "tabla_estadistica.csv")) # Creamos el CSV

    #Graficamos por cada problema (Gráficas compuestas)
    for problem_name in PROBLEMS:
        plot_convergence(
            results, problem_name, os.path.join(args.output_dir, f"convergencia_media_{problem_name}.png")
        )
        plot_best_convergence(
            results, problem_name, os.path.join(args.output_dir, f"convergencia_mejor_{problem_name}.png")
        )

    print("\nExperimento completo. Resultados en:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()