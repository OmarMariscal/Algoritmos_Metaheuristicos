"""
Autor: Mariscal Rodríguez Omar Jesús
Materia: Algoritmos Metaheurísticos
Profesor: Paredes López Ángel Ignasio
Actividad 4 - Algoritmo de Colonia de Abejas Artificiales

Universidad de Guadalajara
Centro Universitario de Ciencias Exactas e Ingenierías

------------------------------------------------------------------------

main.py
Orquesta la Actividad 4 en dos modos:

    --mode tuning (Estretegia heredada de la actividad anterior para la búsqueda de hiperparámetros ideales):
        Barrido de candidatos de (tamaño de colonia SN, multiplicador de
        `limit` respecto a SN x D, máximo de ciclos), con pocas corridas
        por combinación sobre Eggholder. Genera una tabla comparativa para
        elegir manualmente la configuración final


    --mode final (default):
        Corre la configuración elegida con >= 30 corridas independientes
        sobre Eggholder. Genera:
            - Tabla estadística (mejor, peor, media, desviación estándar,
              tasa de éxito).
            - Curva de convergencia Promedio (mejor fitness vs. ciclo,
              promediada sobre las corridas).
            - Una corrida adicional Dedicada (separada de las 30
              estadísticas, con semilla propia reproducible) que captura
              snapshots de la población en 3 momentos (inicial, intermedia,
              final) y los grafica sobre un contorno de la función
              Eggholder para la distribución espacial
            - Gráfica de diversidad poblacional (distancia euclidiana media
              al centroide) vs. ciclo, como información adicional que
              ilustra el efecto de las abejas exploradoras.

Para la modificación:
    - Cambiar selección de vecino de aleatoria a la más cercana euclidiana:
      flag --neighbor-selection nearest_euclidean (o viceversa).
    - Cambiar estrategia de exploradoras (una por ciclo vs. todas las que
      excedan `limit`): flag --scout-strategy all_exceeding.
    - Ambos son solo cambios en ABCConfig; abc_colony.ABCAlgorithm no
      requiere ninguna otra modificación.

Uso:
    python main.py --mode tuning
    python main.py --mode final --colony-size 40 --limit 80 --max-cycles 500
    python main.py --mode final --runs 5   # corrida rápida de prueba
"""

# Importaciones necesarias similares a las actividades pasadas
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import os
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#Importaciones del archivo de configuraciones
from abc_colony import (
    ABCConfig,
    ABCAlgorithm,
    ProblemSpec,
    RunResult,
    is_success,
    make_eggholder,
)

# Constantes para la carpeta de salida y número de veces que se corre el experimento por default
OUTPUT_DIR_DEFAULT = "resultados"
N_RUNS_FINAL_DEFAULT = 30


"""
Utilidades comunes
____________________________________________________________
"""

# Correr una sola vuelta del algoritmo
def run_single(problem: ProblemSpec, config: ABCConfig, seed: int, callback=None) -> RunResult:
    algo = ABCAlgorithm(problem, config)
    return algo.run(rng=np.random.default_rng(seed), callback=callback)

# Correr varias veces el algoritmo mientras se recogen los resultados
def run_many(problem: ProblemSpec, config: ABCConfig, n_runs: int, base_seed: int) -> list[RunResult]:
    rng_seeder = np.random.default_rng(base_seed)
    results = []
    for _ in range(n_runs):
        run_seed = int(rng_seeder.integers(0, 2**31 - 1))
        cfg = dataclasses.replace(config, seed=run_seed)
        results.append(run_single(problem, cfg, run_seed))
    return results

# Cálculo de la curva promedio para las gráficas
def mean_curve(runs: list[RunResult], attr: str) -> np.ndarray:
    curves = [getattr(r, attr) for r in runs]
    max_len = max(len(c) for c in curves)
    padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
    return padded.mean(axis=0)


"""
Modo TUNING
____________________________________________________________
"""
# Contenedor de datos para los resultados del modo Tuning
@dataclass
class TuningRow:
    colony_size: int
    limit: int
    max_cycles: int
    mean_best: float
    std_best: float
    success_rate: float | None
    mean_cycles: float

def parse_int_list(text: str) -> list[int]:
    return [int(v) for v in text.split(",")]

def parse_float_list(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


# Ejecución del modo Tuning
def run_tuning(args: argparse.Namespace) -> list[TuningRow]:
    problem = make_eggholder() # Creación del Contenedor de Datos del Problema EggHolder
    colony_sizes = parse_int_list(args.tuning_colony_sizes) # Número de colonias a int
    limit_multipliers = parse_float_list(args.tuning_limit_multipliers) # Límites que se van a probar
    max_cycles_list = parse_int_list(args.tuning_max_cycles) # Máximo de ciclos por cada combinación

    combos = list(itertools.product(colony_sizes, limit_multipliers, max_cycles_list)) # Todas las combinaciones que el modo probará
    print(f"Modo TUNING: {len(combos)} combinaciones x {args.tuning_runs} corridas = " # Impresiones en consola
          f"{len(combos) * args.tuning_runs} ejecuciones totales sobre Eggholder\n")

    rows: list[TuningRow] = []
    #Bucle del modo tuning
    for i, (sn, limit_mult, max_cycles) in enumerate(combos, start=1):
        limit = max(1, round(limit_mult * sn * problem.n_vars)) # Límite default para el estancamiento
        config = ABCConfig( # Creación de las configuraciones de la run
            colony_size=sn,
            limit=limit,
            max_cycles=max_cycles,
            neighbor_selection=args.neighbor_selection,
            scout_strategy=args.scout_strategy,
            stagnation_cycles=args.stagnation_cycles,
            stagnation_epsilon=args.stagnation_epsilon,
            success_tolerance=args.success_tolerance,
        )

        print(f"[{i}/{len(combos)}] SN={sn} limit={limit} (x{limit_mult} SN*D) max_cycles={max_cycles} "
              f"-> {args.tuning_runs} corridas...", end="", flush=True) 
        t0 = time.time() # Medición del tiempo
        runs = run_many(problem, config, args.tuning_runs, base_seed=args.seed) # Correr la combinación
        elapsed = time.time() - t0

        #Recolectar estadística
        best_values = np.array([r.best_fitness for r in runs])
        successes = [is_success(problem, v, args.success_tolerance) for v in best_values]
        success_rate = float(np.mean(successes)) if all(s is not None for s in successes) else None
        # Agregar la estadística al retorno (para le exportación de la tabla)
        rows.append(
            TuningRow(
                colony_size=sn,
                limit=limit,
                max_cycles=max_cycles,
                mean_best=float(best_values.mean()),
                std_best=float(best_values.std(ddof=1)) if len(best_values) > 1 else 0.0,
                success_rate=success_rate,
                mean_cycles=float(np.mean([r.cycles_run for r in runs])),
            )
        )
        print(f" media={rows[-1].mean_best:.4f} ({elapsed:.1f}s)")

    return rows


#Impresión en consola de los resultados del modo Tuning
def print_tuning_table(rows: list[TuningRow]) -> None:
    header = (f"{'SN':>5} {'limit':>7} {'max_cyc':>8} {'Media mejor':>14} "
              f"{'Desv.Std':>10} {'% Éxito':>9} {'Ciclos media':>13}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        success_str = f"{r.success_rate:.0%}" if r.success_rate is not None else "N/A"
        print(
            f"{r.colony_size:>5} {r.limit:>7} {r.max_cycles:>8} {r.mean_best:>14.4f} "
            f"{r.std_best:>10.4f} {success_str:>9} {r.mean_cycles:>13.1f}"
        )

# Exportación de la tabla de exploración de hiperparámetros a un CSV
def export_tuning_csv(rows: list[TuningRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SN", "limit", "max_cycles", "media_mejor", "desv_std", "tasa_exito", "ciclos_media"])
        for r in rows:
            writer.writerow([
                r.colony_size, r.limit, r.max_cycles, f"{r.mean_best:.6f}", f"{r.std_best:.6f}",
                f"{r.success_rate:.4f}" if r.success_rate is not None else "N/A", f"{r.mean_cycles:.2f}",
            ])
    print(f"\nTabla de sintonización exportada a: {path}")
    print("Recuerda: 'limit' pequeño -> las fuentes se abandonan muy pronto (posible pérdida de "
          "soluciones buenas antes de explotarlas). 'limit' grande -> las exploradoras casi no se "
          "activan y el enjambre puede estancarse en mínimos locales. Revisa la tabla para elegir "
          "un balance y luego corre --mode final con esos valores.")


"""
Modo FINAL
____________________________________________________________
"""

# Contenedor de datos estadísticos
@dataclass
class StatsRow:
    runs: list[RunResult]
    problem_name: str
    best: float
    worst: float
    mean: float
    std: float
    success_rate: float | None
    mean_cycles: float
    last_radio: float

# Cálculos de las estadísticas dado el éxito (si se conoce el mínimo analítico)
def compute_stats(problem: ProblemSpec, runs: list[RunResult], tolerance: float) -> StatsRow:
    values = np.array([r.best_fitness for r in runs])
    successes = [is_success(problem, v, tolerance) for v in values]
    success_rate = float(np.mean(successes)) if all(s is not None for s in successes) else None
    return StatsRow(
        runs= runs,
        problem_name=problem.name,
        best=float(values.min()),
        worst=float(values.max()),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        success_rate=success_rate,
        mean_cycles=float(np.mean([r.cycles_run for r in runs])),
        last_radio=float(runs[-1].radio_list[-1])
    )


# Impresión de la tabla de estadísticas
def print_stats_table(row: StatsRow) -> None:
    header = f"{'Problema':<12} {'Mejor':>12} {'Peor':>12} {'Media':>12} {'Desv.Std':>12} {'% Éxito':>9} {'Ciclos media':>13} {'Último Radio':>12}"
    print("\n" + header)
    print("-" * len(header))
    success_str = f"{row.success_rate:.0%}" if row.success_rate is not None else "N/A"
    print(
        f"{row.problem_name:<12} {row.best:>12.4f} {row.worst:>12.4f} {row.mean:>12.4f} "
        f"{row.std:>12.4f} {success_str:>9} {row.mean_cycles:>13.1f} {row.last_radio:>12.1f}"
    )

    print_radii_runs_table(row)
    # print('=====================================================================')
    # medio_idx = len(row.runs[0].radio_list) // 2
    # print(f'Evolución del radio en una ejecución')
    # print(f'Radio en la Primera Iteración: {row.runs[0].radio_list[0]}')
    # print(f'Radio en la Iteración {medio_idx+1}: {row.runs[0].radio_list[medio_idx]}')
    # print(f'Radio en la Última Iteración: {row.runs[0].radio_list[-1]}')
    

# Exportación de las estadísticas a un CSV
def export_stats_csv(row: StatsRow, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["problema", "mejor", "peor", "media", "desv_std", "tasa_exito", "ciclos_media"])
        writer.writerow([
            row.problem_name, f"{row.best:.6f}", f"{row.worst:.6f}", f"{row.mean:.6f}", f"{row.std:.6f}",
            f"{row.success_rate:.4f}" if row.success_rate is not None else "N/A", f"{row.mean_cycles:.2f}",
        ])
    print(f"\nTabla estadística exportada a: {path}")

#Gráfica de convergenvcia
def plot_convergence(runs: list[RunResult], output_path: str, config_label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    curve = mean_curve(runs, "convergence_best")
    ax.plot(curve, linewidth=1.8, color="tab:blue")
    ax.set_xlabel("Ciclo")
    ax.set_ylabel("Mejor fitness histórico (promedio de las corridas)")
    ax.set_title(f"Curva de convergencia promedio — Eggholder\n({config_label})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de convergencia guardada: {output_path}")


def plot_diversity(runs: list[RunResult], output_path: str, config_label: str) -> None:
    """Gráfica adicional: diversidad poblacional (distancia euclidiana
    media al centroide, normalizada) vs. ciclo para ilustrar el efecto de las
    abejas exploradoras reinyectando diversidad."""
    fig, ax = plt.subplots(figsize=(10, 6))
    curve = mean_curve(runs, "diversity_history")
    ax.plot(curve, linewidth=1.8, color="tab:green")
    ax.set_xlabel("Ciclo")
    ax.set_ylabel("Diversidad euclidiana normalizada (promedio de las corridas)")
    ax.set_title(f"Diversidad poblacional vs. ciclo — Eggholder\n({config_label})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de diversidad poblacional guardada: {output_path}")

def print_radii_runs_table(row: StatsRow):
    """Imprime una tabla detallada donde cada fila es una corrida (run)
    del problema actual, mostrando su radio Inicial, Medio y Final."""
    
    print("\n" + "="*65)
    print(f"EVOLUCIÓN DEL RADIO POR CORRIDA - PROBLEMA: {row.problem_name.upper()}")
    print("="*65)
    print(f"{'No. Corrida':<12} | {'Radio Inicial':<15} | {'Radio Medio':<15} | {'Radio Final':<15}")
    print("-"*65)
    
    # Recorremos cada corrida (run) de forma individual
    for idx, run in enumerate(row.runs, start=1):
        r_list = run.radio_list
        
        # Validar si esta corrida en particular tiene datos
        if not r_list:
            print(f"Run {idx:<8} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
            continue
            
        # Extraer los puntos clave de la lista de esta corrida
        init_val = r_list[0]
        mid_val = r_list[len(r_list) // 2]
        final_val = r_list[-1]
        
        # Imprimir los datos exactos de la corrida con formato científico de 6 decimales
        print(f"Run {idx:<8} | {init_val:<15.6f} | {mid_val:<15.6f} | {final_val:<15.6f}")
        
    print("="*65 + "\n")


def plot_spatial_distribution(
    problem: ProblemSpec, snapshots: list[tuple[int, np.ndarray]], output_path: str, config_label: str
) -> None:
    """Grafica 3 snapshots de la población (inicial, intermedia, final)
    sobre un contorno de la función Eggholder, para visualizar cómo el
    enjambre explora el paisaje multimodal."""
    xs = np.linspace(problem.bounds[0][0], problem.bounds[0][1], 250)
    ys = np.linspace(problem.bounds[1][0], problem.bounds[1][1], 250)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    zz = problem.func(grid_points).reshape(xx.shape)

    labels = ["Generación inicial", "Generación intermedia", "Generación final"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharex=True, sharey=True)

    contour = None
    for ax, (cycle, population), label in zip(axes, snapshots, labels):
        contour = ax.contourf(xx, yy, zz, levels=40, cmap="viridis", alpha=0.8)
        ax.scatter(
            population[:, 0], population[:, 1],
            c="red", edgecolors="white", s=35, linewidths=0.6, zorder=5,
        )
        ax.set_title(f"{label} (ciclo {cycle})")
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
    fig.suptitle(f"Distribución espacial de la población — Eggholder\n({config_label})")
    fig.colorbar(contour, ax=axes, shrink=0.85, label="f(x, y)")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de distribución espacial guardada: {output_path}")


def capture_visualization_run(problem: ProblemSpec, config: ABCConfig, seed: int) -> list[tuple[int, np.ndarray]]:
    """Corre el ABC una vez, capturando la población en cada ciclo, y
    devuelve exactamente 3 snapshots (inicial, intermedia, final) para el
    la distribución espacial pedida. Esta corrida es independiente de
    las N_RUNS estadísticas, para no cargarlas de snapshots innecesarios."""
    history: list[tuple[int, np.ndarray]] = []

    def collector(cycle: int, population: np.ndarray, fitness: np.ndarray) -> None:
        history.append((cycle, population.copy()))

    run_single(problem, config, seed=seed, callback=collector)

    initial = history[0]
    final = history[-1]
    intermediate = history[len(history) // 2]
    return [initial, intermediate, final]

# Ejecución final del algoritmo
def run_final(args: argparse.Namespace) -> None:
    problem = make_eggholder() # Creación del Spec Problem
    config = ABCConfig( # Configuraciones según lo puesto en consola
        colony_size=args.colony_size,
        onlooker_count=args.onlooker_count,
        limit=args.limit,
        max_cycles=args.max_cycles,
        neighbor_selection=args.neighbor_selection,
        scout_strategy=args.scout_strategy,
        stagnation_cycles=args.stagnation_cycles,
        stagnation_epsilon=args.stagnation_epsilon,
        diversity_epsilon=args.diversity_epsilon,
        success_tolerance=args.success_tolerance,
    )
    effective_limit = config.limit if config.limit is not None else config.colony_size * problem.n_vars # Cálculo de límite de estancamiento
    config_label = (
        f"SN={config.colony_size}, limit={effective_limit}, max_cycles={config.max_cycles}, "
        f"vecino={config.neighbor_selection}, exploradoras={config.scout_strategy}"
    )

    print("=" * 70)
    print("Actividad 4 - Algoritmo de Colonia de Abejas Artificiales (modo FINAL)")
    print(f"Corridas estadísticas: {args.runs}")
    print(config_label)
    print("=" * 70)

    print(f"\nEjecutando {args.runs} corridas estadísticas...", end="", flush=True)
    t0 = time.time()
    runs = run_many(problem, config, args.runs, base_seed=args.seed)
    elapsed = time.time() - t0
    print(f" listo ({elapsed:.1f}s)")

    stats_row = compute_stats(problem, runs, args.success_tolerance)
    print_stats_table(stats_row)

    os.makedirs(args.output_dir, exist_ok=True)
    export_stats_csv(stats_row, os.path.join(args.output_dir, "tabla_estadistica_abc.csv"))
    plot_convergence(runs, os.path.join(args.output_dir, "convergencia_eggholder.png"), config_label)
    plot_diversity(runs, os.path.join(args.output_dir, "diversidad_eggholder.png"), config_label)

    # --- Corrida dedicada para distribución espacial (semilla propia, reproducible) ---
    viz_seed = args.seed + 999_999
    print(f"\nCorrida dedicada para distribución espacial (semilla={viz_seed})...", end="", flush=True)
    t0 = time.time()
    snapshots = capture_visualization_run(problem, config, seed=viz_seed)
    elapsed = time.time() - t0
    print(f" listo ({elapsed:.1f}s)")
    plot_spatial_distribution(
        problem, snapshots, os.path.join(args.output_dir, "distribucion_espacial_eggholder.png"), config_label
    )

    print("\nExperimento final completo. Resultados en:", os.path.abspath(args.output_dir))


"""
CLI para cambiar estretegias directamente en consola sin modificar código
____________________________________________________________
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actividad 4 - ABC: tuning y experimento final sobre Eggholder")
    parser.add_argument("--mode", type=str, default="final", choices=["tuning", "final"])

    # Comunes
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbor-selection", type=str, default="random",
                         choices=["random", "nearest_euclidean"])
    parser.add_argument("--scout-strategy", type=str, default="single_worst",
                         choices=["single_worst", "all_exceeding"])
    parser.add_argument("--stagnation-cycles", type=int, default=60)
    parser.add_argument("--stagnation-epsilon", type=float, default=1e-6)
    parser.add_argument("--diversity-epsilon", type=float, default=0.0)
    parser.add_argument("--success-tolerance", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)

    # Modo final
    parser.add_argument("--runs", type=int, default=N_RUNS_FINAL_DEFAULT)
    parser.add_argument("--colony-size", type=int, default=40)
    parser.add_argument("--onlooker-count", type=int, default=None,
                         help="Default: igual a --colony-size")
    parser.add_argument("--limit", type=int, default=None,
                         help="Default: colony_size * n_vars (regla SN x D)")
    parser.add_argument("--max-cycles", type=int, default=500)

    # Modo tuning
    parser.add_argument("--tuning-runs", type=int, default=10)
    parser.add_argument("--tuning-colony-sizes", type=str, default="20,40,80")
    parser.add_argument("--tuning-limit-multipliers", type=str, default="0.5,1,2",
                         help="limit = multiplicador * SN * D")
    parser.add_argument("--tuning-max-cycles", type=str, default="200,400")

    return parser.parse_args()

# Función main
def main() -> None:
    args = parse_args() # Argumentos de consola

    if args.mode == "tuning": # Para el modo tuning
        rows = run_tuning(args)
        print_tuning_table(rows)
        os.makedirs(args.output_dir, exist_ok=True)
        export_tuning_csv(rows, os.path.join(args.output_dir, "tabla_sintonizacion_abc.csv"))
    else: # Para el modo final
        run_final(args)


if __name__ == "__main__":
    main()
