"""
Autor: Mariscal Rodríguez Omar Jesús
Materia: Algoritmos Metaheurísticos
Profesor: Paredes López Ángel Ignasio
Actividad 3 - Algoritmo Genético Continuo

Universidad de Guadalajara
Centro Universitario de Ciencias Exactas e Ingenierías

------------------------------------------------------------------------


main.py
=======
Orquesta la Actividad 3 en dos modos:

    --mode tuning (default de exploración):
        Barrido de candidatos de (población, Pc, Pm) sobre Ackley 2D (mide
        capacidad de evadir mínimos locales / exploración) y Esfera 4D (mide
        precisión / explotación), con pocas corridas por combinación. Genera
        una tabla comparativa para elegir manualmente la configuración final.

    --mode final:
        Corre la configuración elegida (pasada por CLI) con >= 30 corridas
        independientes sobre los 3 problemas de la actividad: Ackley 2D,
        Esfera 4D y Esfera 10D. Genera tabla estadística (mejor, peor, media,
        desv. std, tasa de éxito), curvas de convergencia promedio por
        problema, y una gráfica comparativa Esfera 4D vs 10D (escalabilidad).


Para modificaciones al patrón de estrategia igual que la actividad anterior::
    - Cambiar operador de cruza/mutación/selección/manejo de límites: son
      flags de CLI que solo cambian GAConfig; ga_continuo.GeneticAlgorithm
      no requiere ningún otro cambio.
    - Agregar un operador nuevo: se registra en el diccionario correspondiente
      dentro de ga_continuo.py (SELECTION_OPERATORS / CROSSOVER_OPERATORS /
      MUTATION_OPERATORS / BOUNDARY_HANDLERS).

Ejemplos de Uso:
    python main.py --mode tuning
    python main.py --mode final --population 60 --crossover-prob 0.9 --mutation-prob 0.1
    python main.py --mode final --runs 5   # corrida rápida de prueba
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools # Importación para hacer más eficientes los bucles para el modo de exploración o modo tuning
import os
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ga_continuo import (
    GAConfig,
    GeneticAlgorithm,
    ProblemSpec,
    RunResult,
    is_success,
    make_ackley_2d,
    make_sphere,
)


OUTPUT_DIR_DEFAULT = "resultados"
N_RUNS_FINAL_DEFAULT = 30


"""
Utilidades comunes
____________________________________________________________
"""

# Crear los problemas con los contenedores de los problemas con las funciones auxiliares
def build_problems() -> dict[str, ProblemSpec]:
    return {
        "ackley_2d": make_ackley_2d(),
        "sphere_4d": make_sphere(4),
        "sphere_10d": make_sphere(10),
    }

# Correr una sola corrida
def run_single(problem: ProblemSpec, config: GAConfig, seed: int) -> RunResult:
    ga = GeneticAlgorithm(problem, config)
    return ga.run(rng=np.random.default_rng(seed))

# Correr varias veces el algoritmo
def run_many(problem: ProblemSpec, config: GAConfig, n_runs: int, base_seed: int) -> list[RunResult]:
    rng_seeder = np.random.default_rng(base_seed)
    results = []
    for _ in range(n_runs):
        run_seed = int(rng_seeder.integers(0, 2**31 - 1))
        cfg = dataclasses.replace(config, seed=run_seed)
        results.append(run_single(problem, cfg, run_seed))
    return results


def mean_convergence_curve(runs: list[RunResult], kind: str = "mean") -> np.ndarray:
    curves = [r.convergence_mean if kind == "mean" else r.convergence_best for r in runs]
    max_len = max(len(c) for c in curves)
    padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
    return padded.mean(axis=0)


"""
Modo TUNING
____________________________________________________________
"""

@dataclass # Contenedor de datos para la configuración del modo tuning
class TuningRow:
    population: int
    crossover_prob: float
    mutation_prob: float
    problem_name: str
    mean_best: float
    std_best: float
    success_rate: float | None
    mean_generations: float

# Para la línea de comandos, separadores por comas de reales
def parse_float_list(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]

# Para la lista de comandos, separadores por comas de enteros
def parse_int_list(text: str) -> list[int]:
    return [int(v) for v in text.split(",")]

# Ejecutar el modo tuning
def run_tuning(args: argparse.Namespace) -> list[TuningRow]:
    populations = parse_int_list(args.tuning_populations) # El número de poblaciones a probar
    pcs = parse_float_list(args.tuning_pc) # Probabilidad de cruces a probar
    pms = parse_float_list(args.tuning_pm) # Probabiliaddes de mutaciones a probar

    tuning_problems = {
        "ackley_2d": make_ackley_2d(),        # exploración: evadir mínimos locales
        "sphere_4d": make_sphere(4),          # explotación: precisión
    }

    rows: list[TuningRow] = [] # Lista de resultados que almacenaremos
    combos = list(itertools.product(populations, pcs, pms)) # Lista de combinaciones que almacenarmos utilizando itertolls
    total = len(combos) * len(tuning_problems) # TOtal de combinaciones x problemas
    i = 0 # Contador 

    print(f"Modo TUNING: {len(combos)} combinaciones x {len(tuning_problems)} problemas "
          f"x {args.tuning_runs} corridas = {total * args.tuning_runs} ejecuciones totales\n")

    for pop, pc, pm in combos: # Bucle principal que se recorre por cada combinación
        base_config = GAConfig( # Creación de la configuración de un combo
            population_size=pop,
            crossover_prob=pc,
            mutation_prob=pm,
            max_generations=args.max_generations,
            selection=args.selection,
            crossover=args.crossover,
            mutation=args.mutation,
            boundary_handling=args.boundary_handling,
            elitism=args.elitism,
            stagnation_generations=args.stagnation_generations,
            stagnation_epsilon=args.stagnation_epsilon,
            success_tolerance=args.success_tolerance,
        )

        for problem_name, problem in tuning_problems.items(): # Bucle para las impresiones de cada problema para el combo
            i += 1
            print(f"[{i}/{len(combos) * len(tuning_problems)}] pop={pop} Pc={pc} Pm={pm} "
                  f"| {problem_name} -> {args.tuning_runs} corridas...", end="", flush=True)
            t0 = time.time() # Medición del tiempo 
            runs = run_many(problem, base_config, args.tuning_runs, base_seed=args.seed) # Correr tantas veces como pruebas pedidas
            elapsed = time.time() - t0 # Tiempo de finalización

            best_values = np.array([r.best_fitness for r in runs]) # Mejores valores obtenidos
            successes = [is_success(problem, v, args.success_tolerance) for v in best_values] # Cuantos tuvieron éxito en relación al óptimo conocido
            success_rate = float(np.mean(successes)) if all(s is not None for s in successes) else None # Porcentaje de casos exitosos

            rows.append( # Agregar los resultados del combo/problema
                TuningRow(
                    population=pop,
                    crossover_prob=pc,
                    mutation_prob=pm,
                    problem_name=problem_name,
                    mean_best=float(best_values.mean()),
                    std_best=float(best_values.std(ddof=1)) if len(best_values) > 1 else 0.0,
                    success_rate=success_rate,
                    mean_generations=float(np.mean([r.generations_run for r in runs])),
                )
            )
            print(f" media={rows[-1].mean_best:.5f} ({elapsed:.1f}s)")

    return rows

#Impresion de resultados de tuning
def print_tuning_table(rows: list[TuningRow]) -> None:
    header = (f"{'Pob':>5} {'Pc':>5} {'Pm':>6} {'Problema':<12} "
              f"{'Media mejor':>12} {'Desv.Std':>10} {'% Éxito':>9} {'Gen. media':>10}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        success_str = f"{r.success_rate:.0%}" if r.success_rate is not None else "N/A"
        print(
            f"{r.population:>5} {r.crossover_prob:>5.2f} {r.mutation_prob:>6.2f} {r.problem_name:<12} "
            f"{r.mean_best:>12.6f} {r.std_best:>10.6f} {success_str:>9} {r.mean_generations:>10.1f}"
        )

#Exportación de la tabla de prueba
def export_tuning_csv(rows: list[TuningRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["poblacion", "Pc", "Pm", "problema", "media_mejor", "desv_std", "tasa_exito", "gen_media"])
        for r in rows:
            writer.writerow([
                r.population, r.crossover_prob, r.mutation_prob, r.problem_name,
                f"{r.mean_best:.10f}", f"{r.std_best:.10f}",
                f"{r.success_rate:.4f}" if r.success_rate is not None else "N/A",
                f"{r.mean_generations:.2f}",
            ])
    print(f"\nTabla de sintonización exportada a: {path}")



"""
Modo FINAL
____________________________________________________________
"""

#Contenedor de datos para la corrida del algoritmo
@dataclass
class StatsRow:
    problem_name: str
    best: float
    worst: float
    mean: float
    std: float
    success_rate: float | None
    mean_generations: float

# Calcular estadísticas
def compute_stats(problem: ProblemSpec, runs: list[RunResult], tolerance: float) -> StatsRow:
    values = np.array([r.best_fitness for r in runs])
    successes = [is_success(problem, v, tolerance) for v in values]
    success_rate = float(np.mean(successes)) if all(s is not None for s in successes) else None
    return StatsRow(
        problem_name=problem.name,
        best=float(values.min()),
        worst=float(values.max()),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        success_rate=success_rate,
        mean_generations=float(np.mean([r.generations_run for r in runs])),
    )

# Impresiones
def print_stats_table(rows: list[StatsRow]) -> None:
    header = f"{'Problema':<12} {'Mejor':>12} {'Peor':>12} {'Media':>12} {'Desv.Std':>12} {'% Éxito':>9} {'Gen.media':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        success_str = f"{r.success_rate:.0%}" if r.success_rate is not None else "N/A"
        print(
            f"{r.problem_name:<12} {r.best:>12.6f} {r.worst:>12.6f} {r.mean:>12.6f} "
            f"{r.std:>12.6f} {success_str:>9} {r.mean_generations:>10.1f}"
        )

# Exportar la tabla de resultados
def export_stats_csv(rows: list[StatsRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["problema", "mejor", "peor", "media", "desv_std", "tasa_exito", "gen_media"])
        for r in rows:
            writer.writerow([
                r.problem_name, f"{r.best:.10f}", f"{r.worst:.10f}", f"{r.mean:.10f}", f"{r.std:.10f}",
                f"{r.success_rate:.4f}" if r.success_rate is not None else "N/A",
                f"{r.mean_generations:.2f}",
            ])
    print(f"\nTabla estadística exportada a: {path}")

#Graficar convergencia
def plot_convergence(problem_name: str, runs: list[RunResult], output_path: str, config_label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_curve = mean_convergence_curve(runs, kind="mean")
    best_curve = mean_convergence_curve(runs, kind="best")

    ax.plot(mean_curve, label="Fitness medio poblacional (promedio de corridas)", linewidth=1.5)
    ax.plot(best_curve, label="Mejor fitness (promedio de corridas)", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Convergencia promedio — {problem_name}\n({config_label})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de convergencia guardada: {output_path}")

# Graficar la convergencia promedio
def plot_sphere_scalability(
    runs_4d: list[RunResult], runs_10d: list[RunResult], output_path: str, config_label: str
) -> None:
    """Gráfica comparativa de convergencia promedio: Esfera 4D vs 10D"""
    fig, ax = plt.subplots(figsize=(10, 6))

    curve_4d = mean_convergence_curve(runs_4d, kind="best")
    curve_10d = mean_convergence_curve(runs_10d, kind="best")

    ax.plot(curve_4d, label="Esfera 4D", linewidth=1.8)
    ax.plot(curve_10d, label="Esfera 10D", linewidth=1.8)
    ax.set_yscale("log")

    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor fitness (escala log, promedio de corridas)")
    ax.set_title(f"Escalabilidad: Esfera 4D vs 10D\n({config_label})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de escalabilidad (4D vs 10D) guardada: {output_path}")

# Algoritmo final con los hiperparámetros elegidos
def run_final(args: argparse.Namespace) -> None:
    config = GAConfig( # Creación de las configuraciones
        population_size=args.population,
        max_generations=args.max_generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        elitism=args.elitism,
        selection=args.selection,
        tournament_k=args.tournament_k,
        crossover=args.crossover,
        crossover_alpha=args.crossover_alpha,
        mutation=args.mutation,
        mutation_sigma_frac=args.mutation_sigma_frac,
        boundary_handling=args.boundary_handling,
        stagnation_generations=args.stagnation_generations,
        stagnation_epsilon=args.stagnation_epsilon,
        diversity_epsilon=args.diversity_epsilon,
        success_tolerance=args.success_tolerance,
    )
    config_label = ( # Impresiones 
        f"pop={config.population_size}, Pc={config.crossover_prob}, Pm={config.mutation_prob}, "
        f"sel={config.selection}, cruza={config.crossover}, mut={config.mutation}, "
        f"límites={config.boundary_handling}"
    )

    print("=" * 70)
    print("Actividad 3 - Algoritmo Genético Continuo (modo FINAL)")
    print(f"Corridas por problema: {args.runs}")
    print(config_label)
    print("=" * 70)

    problems = build_problems()
    all_runs: dict[str, list[RunResult]] = {}
    stats_rows: list[StatsRow] = []

    for name, problem in problems.items(): # Bublce principal por cada problema 
        print(f"\nEjecutando {name} ({args.runs} corridas)...", end="", flush=True)
        t0 = time.time()
        runs = run_many(problem, config, args.runs, base_seed=args.seed)
        elapsed = time.time() - t0
        print(f" listo ({elapsed:.1f}s)")

        all_runs[name] = runs
        stats_rows.append(compute_stats(problem, runs, args.success_tolerance))
    #Imprimir los resultados
    print_stats_table(stats_rows)
    #Exportar la tabla
    os.makedirs(args.output_dir, exist_ok=True)
    export_stats_csv(stats_rows, os.path.join(args.output_dir, "tabla_estadistica_continuo.csv"))
    
    for name, runs in all_runs.items():
        plot_convergence(
            name, runs, os.path.join(args.output_dir, f"convergencia_{name}.png"), config_label
        )

    plot_sphere_scalability(
        all_runs["sphere_4d"], all_runs["sphere_10d"],
        os.path.join(args.output_dir, "escalabilidad_esfera_4d_vs_10d.png"),
        config_label,
    )

    print("\nExperimento final completo. Resultados en:", os.path.abspath(args.output_dir))


"""
CLI / punto de entrada
____________________________________________________________
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actividad 3 - GA Continuo: tuning y experimento final")
    parser.add_argument("--mode", type=str, default="final", choices=["tuning", "final"])

    # Comunes
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-generations", type=int, default=300)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--selection", type=str, default="tournament", choices=["roulette", "tournament"])
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--crossover", type=str, default="blx_alpha",
                         choices=["arithmetic", "blx_alpha", "uniform_real"])
    parser.add_argument("--crossover-alpha", type=float, default=0.5)
    parser.add_argument("--mutation", type=str, default="gaussian", choices=["gaussian", "uniform_reset"])
    parser.add_argument("--mutation-sigma-frac", type=float, default=0.1)
    parser.add_argument("--boundary-handling", type=str, default="clip",
                         choices=["clip", "reflect", "resample"])
    parser.add_argument("--stagnation-generations", type=int, default=30)
    parser.add_argument("--stagnation-epsilon", type=float, default=1e-8)
    parser.add_argument("--diversity-epsilon", type=float, default=0.0)
    parser.add_argument("--success-tolerance", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)

    # Modo final
    parser.add_argument("--runs", type=int, default=N_RUNS_FINAL_DEFAULT)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--crossover-prob", type=float, default=0.9)
    parser.add_argument("--mutation-prob", type=float, default=0.1)

    # Modo tuning
    parser.add_argument("--tuning-runs", type=int, default=10)
    parser.add_argument("--tuning-populations", type=str, default="30,60,100")
    parser.add_argument("--tuning-pc", type=str, default="0.7,0.85,0.95")
    parser.add_argument("--tuning-pm", type=str, default="0.02,0.1,0.2")

    return parser.parse_args()

# Función Principal
def main() -> None:
    args = parse_args()

    if args.mode == "tuning": # Si se hará el modo de prueba o tuning
        rows = run_tuning(args)
        print_tuning_table(rows)
        os.makedirs(args.output_dir, exist_ok=True)
        export_tuning_csv(rows, os.path.join(args.output_dir, "tabla_sintonizacion.csv"))
    else:
        run_final(args) # Si el modo es el final


if __name__ == "__main__":
    main()