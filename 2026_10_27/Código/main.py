"""
Autor: Mariscal Rodríguez Omar Jesús
Materia: Algoritmos Metaheurísticos
Profesor: Paredes López Ángel Ignasio
Actividad 5 - Optimización por Enjambre de Partículas (PSO)

Universidad de Guadalajara
Centro Universitario de Ciencias Exactas e Ingenierías

------------------------------------------------------------------------

main.py

Orquesta la Actividad 5 (PSO) en dos modos:

    --mode tuning:
        Barrido de (tamaño del enjambre N, peso de inercia w, coeficiente
        cognitivo c1, coeficiente social c2). Cada combinación se evalúa en
        AMBAS funciones (Rosenbrock y Michalewicz) con pocas corridas, y se
        genera un ranking de configuraciones COMUNES a ambas (deliverable
        "Sintonización de Parámetros"). El ranking ordena por tasa de éxito
        promedio y, como desempate, por la suma de rangos de la media del
        mejor fitness en cada función (métrica sin escala, comparable entre
        funciones con magnitudes muy distintas).

    --mode final (default):
        Corre la configuración elegida con >= 30 corridas independientes
        sobre cada función. Genera, por función:
            - Tabla estadística (mejor, peor, media, desv. estándar, éxito).
            - Curva de convergencia promedio (mejor global y fitness medio
              del enjambre vs. iteración).
            - Scatter de las partículas en 3 momentos (inicial, intermedia,
              final) sobre el contorno de la función, con una corrida
              DEDICADA (semilla propia reproducible, separada de las 30).
            - Gráfica de diversidad euclidiana vs. iteración (extra).

Para modificar en vivo (evaluación presencial), basta cambiar un flag:
    --topology global | ring | euclidean_knn      (--neighborhood-k)
    --inertia-schedule constant | linear_decay    (--w, --w-start, --w-end)
    --boundary-handling clip | reflect | random_reset
    Todos solo modifican PSOConfig; pso.PSO no requiere otros cambios.

Uso:
    python main.py --mode tuning
    python main.py --mode final --swarm-size 40 --w 0.7 --c1 1.5 --c2 1.5
    python main.py --mode final --runs 5   # corrida rápida de prueba
"""

# Importaciones necesarias
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

from pso import (
    PSO,
    PSOConfig,
    ProblemSpec,
    RunResult,
    is_success,
    make_michalewicz,
    make_rosenbrock,
)

# Ajustes por defecto
OUTPUT_DIR_DEFAULT = "resultados"
N_RUNS_FINAL_DEFAULT = 30


"""
Utilidades comunes
____________________________________________________________
"""

# Contruir los problemas pedidos
def build_problems() -> dict[str, ProblemSpec]:
    return {"rosenbrock": make_rosenbrock(2), "michalewicz": make_michalewicz(2, m=10)}

# Correr una vez el elgoritmo
def run_single(problem: ProblemSpec, config: PSOConfig, seed: int, callback=None) -> RunResult:
    return PSO(problem, config).run(rng=np.random.default_rng(seed), callback=callback)

# Correr varias veces el algoritmo
def run_many(problem: ProblemSpec, config: PSOConfig, n_runs: int, base_seed: int) -> list[RunResult]:
    seeder = np.random.default_rng(base_seed)
    results = []
    for _ in range(n_runs):
        run_seed = int(seeder.integers(0, 2**31 - 1))
        results.append(run_single(problem, dataclasses.replace(config, seed=run_seed), run_seed))
    return results

# Promedio de la curba
def mean_curve(runs: list[RunResult], attr: str) -> np.ndarray:
    """Promedia una curva por iteración entre corridas; las corridas más
    cortas (paro débil) se extienden con su último valor."""
    curves = [getattr(r, attr) for r in runs]
    max_len = max(len(c) for c in curves)
    return np.array([c + [c[-1]] * (max_len - len(c)) for c in curves]).mean(axis=0)

# De consola a real
def parse_float_list(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]

#De consola a entero
def parse_int_list(text: str) -> list[int]:
    return [int(v) for v in text.split(",")]

# Construcción desde consola del PSO Config
def config_from_args(args: argparse.Namespace, **overrides) -> PSOConfig:
    """Construye un PSOConfig desde los flags de CLI, con overrides opcionales."""
    base = dict(
        swarm_size=args.swarm_size,
        max_iterations=args.max_iterations,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        topology=args.topology,
        neighborhood_k=args.neighborhood_k,
        inertia_schedule=args.inertia_schedule,
        w_start=args.w_start,
        w_end=args.w_end,
        boundary_handling=args.boundary_handling,
        v_max_frac=args.v_max_frac,
        stagnation_iterations=args.stagnation_iterations,
        stagnation_epsilon=args.stagnation_epsilon,
        diversity_epsilon=args.diversity_epsilon,
        success_tolerance=args.success_tolerance,
    )
    base.update(overrides)
    return PSOConfig(**base)

# Función para impresiones de hiperparámetros
def config_label(cfg: PSOConfig) -> str:
    w_txt = f"w={cfg.w}" if cfg.inertia_schedule == "constant" else f"w:{cfg.w_start}->{cfg.w_end}"
    return (f"N={cfg.swarm_size}, {w_txt}, c1={cfg.c1}, c2={cfg.c2}, topología={cfg.topology}"
            f"(k={cfg.neighborhood_k}), fronteras={cfg.boundary_handling}")


"""
Modo TUNING
____________________________________________________________
"""
#Contenedor de datos del modo tuning
@dataclass
class TuningRow:
    swarm_size: int
    w: float
    c1: float
    c2: float
    problem: str
    mean_best: float
    std_best: float
    success_rate: float | None

# Corrida del modo tuning
def run_tuning(args: argparse.Namespace) -> list[TuningRow]:
    problems = build_problems() # Construcción de los problemas
    sizes = parse_int_list(args.tuning_swarm_sizes) # Pasamos a enterlos los tamaños
    ws = parse_float_list(args.tuning_w) # Peso de inercia
    c1s = parse_float_list(args.tuning_c1) # Peso del factor cognitivo
    c2s = parse_float_list(args.tuning_c2) # Peso del factor social

    # Si el usuario elige linear_decay, el valor de w que se barre no se usa (se ignoran los valores del barrido y se usan w_start/w_end en su lugar).
    if args.inertia_schedule != "constant":
        print("AVISO: el tuning de 'w' asume inercia constante; con linear_decay el valor de w del "
              "barrido no se usa (se usan --w-start/--w-end).\n")

    combos = list(itertools.product(sizes, ws, c1s, c2s)) # Lista de combinaciones
    total_runs = len(combos) * len(problems) * args.tuning_runs # Cálculo del total de runs necerias
    print(f"Modo TUNING: {len(combos)} combinaciones x {len(problems)} funciones x "
          f"{args.tuning_runs} corridas = {total_runs} ejecuciones\n")

    rows: list[TuningRow] = []
    t_start = time.time() # Tiempo para logs
    for i, (n, w, c1, c2) in enumerate(combos, start=1): # Bucle principal
        cfg = config_from_args(args, swarm_size=n, w=w, c1=c1, c2=c2) # Configuraciones esenciales
        for pname, problem in problems.items(): # Por combinación
            runs = run_many(problem, cfg, args.tuning_runs, base_seed=args.seed) # Correr el algoritmo
            vals = np.array([r.best_fitness for r in runs]) # Obtner los mejores fitness
            succ = [is_success(problem, v, args.success_tolerance) for v in vals] # Calcular el porcentaje de éxito basado en el mínimo conocido
            rows.append(TuningRow( # Estadística final
                swarm_size=n, w=w, c1=c1, c2=c2, problem=pname,
                mean_best=float(vals.mean()),
                std_best=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                success_rate=float(np.mean(succ)) if all(s is not None for s in succ) else None,
            ))
        if i % 10 == 0 or i == len(combos):
            print(f"  [{i}/{len(combos)}] combinaciones evaluadas ({time.time() - t_start:.0f}s)") # Impresiones finales
    return rows

#Contenedor de datos del modo tuning
@dataclass
class CombinedRow:
    swarm_size: int
    w: float
    c1: float
    c2: float
    mean_best: dict[str, float]
    success: dict[str, float]
    combined_success: float
    rank_sum: float


def combine_tuning_rows(rows: list[TuningRow]) -> list[CombinedRow]:
    """Agrupa por configuración y calcula el ranking común a ambas funciones."""
    groups: dict[tuple, dict[str, TuningRow]] = {}
    for r in rows:
        groups.setdefault((r.swarm_size, r.w, r.c1, r.c2), {})[r.problem] = r

    problems = sorted({r.problem for r in rows})
    # Rango (1 = mejor) de cada configuración por función, según la media del mejor fitness
    ranks: dict[str, dict[tuple, int]] = {}
    for p in problems:
        ordered = sorted(groups, key=lambda k: groups[k][p].mean_best)
        ranks[p] = {k: pos + 1 for pos, k in enumerate(ordered)}

    combined = [] # Combinaciones
    for key, per_problem in groups.items(): # Por combinación
        success = {p: (per_problem[p].success_rate or 0.0) for p in problems}
        combined.append(CombinedRow(
            swarm_size=key[0], w=key[1], c1=key[2], c2=key[3],
            mean_best={p: per_problem[p].mean_best for p in problems},
            success=success,
            combined_success=float(np.mean(list(success.values()))),
            rank_sum=float(sum(ranks[p][key] for p in problems)),
        ))
    combined.sort(key=lambda c: (-c.combined_success, c.rank_sum))
    return combined

#Función auxiliar de impresión del modo tuning
def print_tuning_ranking(combined: list[CombinedRow], top: int = 15) -> None:
    problems = list(combined[0].mean_best)
    header = f"{'Pos':>3} {'N':>4} {'w':>5} {'c1':>5} {'c2':>5} {'Éxito comb.':>12} "
    header += " ".join(f"{'Éxito ' + p[:5]:>13} {'Media ' + p[:5]:>13}" for p in problems)
    print(f"\nTop {top} configuraciones comunes (orden: éxito combinado, luego suma de rangos)")
    print(header)
    print("-" * len(header))
    for pos, c in enumerate(combined[:top], start=1):
        line = f"{pos:>3} {c.swarm_size:>4} {c.w:>5.2f} {c.c1:>5.2f} {c.c2:>5.2f} {c.combined_success:>11.0%} "
        line += " ".join(f"{c.success[p]:>13.0%} {c.mean_best[p]:>13.6f}" for p in problems)
        print(line)

#Función auxiliar para la exportación de resultados del modo tuning para el reporte
def export_tuning_csv(combined: list[CombinedRow], path: str) -> None:
    problems = list(combined[0].mean_best)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["posicion", "N", "w", "c1", "c2", "exito_combinado", "suma_rangos"]
        for p in problems:
            header += [f"exito_{p}", f"media_mejor_{p}"]
        writer.writerow(header)
        for pos, c in enumerate(combined, start=1):
            row = [pos, c.swarm_size, c.w, c.c1, c.c2, f"{c.combined_success:.4f}", f"{c.rank_sum:.0f}"]
            for p in problems:
                row += [f"{c.success[p]:.4f}", f"{c.mean_best[p]:.8f}"]
            writer.writerow(row)
    print(f"\nTabla completa de sintonización exportada a: {path}")
    print("Elige una configuración con buen éxito combinado y corre --mode final con esos valores.")

"""
Modo FINAL
____________________________________________________________
"""

# Contenedor de datos para el modo final 
@dataclass
class StatsRow:
    problem: str
    best: float
    worst: float
    mean: float
    std: float
    success_rate: float | None
    mean_iterations: float

# Computar estadísticas del modo fincla
def compute_stats(problem: ProblemSpec, runs: list[RunResult], tolerance: float) -> StatsRow:
    vals = np.array([r.best_fitness for r in runs])
    succ = [is_success(problem, v, tolerance) for v in vals]
    return StatsRow(
        problem=problem.name,
        best=float(vals.min()),
        worst=float(vals.max()),
        mean=float(vals.mean()),
        std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        success_rate=float(np.mean(succ)) if all(s is not None for s in succ) else None,
        mean_iterations=float(np.mean([r.iterations_run for r in runs])),
    )

# Impresión del modo final
def print_stats_table(rows: list[StatsRow]) -> None:
    header = (f"{'Problema':<16} {'Mejor':>14} {'Peor':>14} {'Media':>14} {'Desv.Std':>14} "
              f"{'% Éxito':>8} {'Iter. media':>12}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        s = f"{r.success_rate:.0%}" if r.success_rate is not None else "N/A"
        print(f"{r.problem:<16} {r.best:>14.8f} {r.worst:>14.8f} {r.mean:>14.8f} {r.std:>14.8f} "
              f"{s:>8} {r.mean_iterations:>12.1f}")

#Exportar a csv la ejecución final
def export_stats_csv(rows: list[StatsRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["problema", "mejor", "peor", "media", "desv_std", "tasa_exito", "iter_media"])
        for r in rows:
            writer.writerow([
                r.problem, f"{r.best:.10f}", f"{r.worst:.10f}", f"{r.mean:.10f}", f"{r.std:.10f}",
                f"{r.success_rate:.4f}" if r.success_rate is not None else "N/A", f"{r.mean_iterations:.2f}",
            ])
    print(f"\nTabla estadística exportada a: {path}")

# Gráfica de la convergencia
def plot_convergence(problem: ProblemSpec, runs: list[RunResult], path: str, label: str) -> None:
    """Mejor global promedio y fitness medio del enjambre promedio vs. iteración."""
    best_curve = mean_curve(runs, "convergence_best")
    mean_curve_swarm = mean_curve(runs, "convergence_mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best_curve, linewidth=1.8, label="Mejor global (promedio de corridas)")
    ax.plot(mean_curve_swarm, linewidth=1.5, linestyle="--", label="Fitness medio del enjambre (promedio de corridas)")
    if problem.known_optimum is not None:
        ax.axhline(problem.known_optimum, color="black", linestyle=":", linewidth=1, label="Óptimo global")
    if min(best_curve.min(), mean_curve_swarm.min()) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Convergencia promedio — {problem.name}\n({label})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de convergencia guardada: {path}")

# Graficar diversidad
def plot_diversity(problem: ProblemSpec, runs: list[RunResult], path: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mean_curve(runs, "diversity_history"), linewidth=1.8, color="tab:green")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Diversidad euclidiana normalizada (promedio)")
    ax.set_title(f"Diversidad del enjambre vs. iteración — {problem.name}\n({label})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de diversidad guardada: {path}")

# Captura de resumen en momentos específicos de la corrida
def capture_snapshots(problem: ProblemSpec, config: PSOConfig, seed: int) -> list[tuple[int, np.ndarray]]:
    """Corrida dedicada (independiente de las N estadísticas) que captura las
    posiciones del enjambre en cada iteración y devuelve 3 snapshots:
    inicial, intermedia (mitad de las iteraciones realmente ejecutadas) y final."""
    history: list[tuple[int, np.ndarray]] = []
    run_single(problem, config, seed, callback=lambda it, pos, fit: history.append((it, pos.copy())))
    return [history[0], history[len(history) // 2], history[-1]]

# Graficar la distribución espacial
def plot_spatial_distribution(
    problem: ProblemSpec, snapshots: list[tuple[int, np.ndarray]], path: str, label: str
) -> None:
    """Scatter de las partículas en 3 momentos sobre el contorno de la función."""
    (x_lo, x_hi), (y_lo, y_hi) = problem.bounds[0], problem.bounds[1]
    xs, ys = np.linspace(x_lo, x_hi, 300), np.linspace(y_lo, y_hi, 300)
    xx, yy = np.meshgrid(xs, ys)
    zz = problem.func(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Rosenbrock abarca varios órdenes de magnitud: contorno en escala log para ver el valle
    use_log = zz.min() >= 0
    z_plot = np.log10(zz + 1.0) if use_log else zz
    cbar_label = "log10(f + 1)" if use_log else "f(x, y)"

    titles = ["Generación inicial", "Generación intermedia", "Generación final"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharex=True, sharey=True)
    contour = None
    for ax, (it, pos), title in zip(axes, snapshots, titles):
        contour = ax.contourf(xx, yy, z_plot, levels=40, cmap="viridis", alpha=0.85)
        # El óptimo se dibuja DEBAJO de las partículas (zorder menor): al final de la
        # corrida las partículas se concentran sobre él y, si no, quedarían tapadas.
        if problem.optimum_position is not None:
            ax.scatter(*problem.optimum_position[:2], marker="*", c="gold", edgecolors="black",
                       s=420, zorder=4, label="Óptimo global")
        ax.scatter(pos[:, 0], pos[:, 1], c="red", edgecolors="white", s=22, linewidths=0.5, zorder=6,
                   label="Partículas")
        ax.set_title(f"{title} (iteración {it})")
        ax.set_xlabel("x")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Distribución espacial del enjambre — {problem.name}\n({label})")
    fig.colorbar(contour, ax=axes, shrink=0.85, label=cbar_label)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Gráfica de distribución espacial guardada: {path}")

# Correr el modo final
def run_final(args: argparse.Namespace) -> None:
    config = config_from_args(args) # Cargar configuraciones
    label = config_label(config) # Callable del headder de hiper-parámetros
    problems = build_problems() # Construcción de problemas

    # Impresión Inicial
    print("=" * 78)
    print("Actividad 5 - Optimización por Enjambre de Partículas (modo FINAL)")
    print(f"Corridas por función: {args.runs}")
    print(label)
    print("=" * 78)

    os.makedirs(args.output_dir, exist_ok=True) # Carpeta de exportación
    stats_rows: list[StatsRow] = [] # Estadísticas

    for key, problem in problems.items(): # Blucle principal
        print(f"\nEjecutando {problem.name} ({args.runs} corridas)...", end="", flush=True) # Inicio del modo final
        t0 = time.time() # Medición del tiempo
        runs = run_many(problem, config, args.runs, base_seed=args.seed) # Correr varias veces el algoritmo
        print(f" listo ({time.time() - t0:.1f}s)") # Final del tiempo
        stats_rows.append(compute_stats(problem, runs, args.success_tolerance)) # Cálculo de estadísticas

        plot_convergence(problem, runs, os.path.join(args.output_dir, f"convergencia_{key}.png"), label) # Graficar Convergencia
        plot_diversity(problem, runs, os.path.join(args.output_dir, f"diversidad_{key}.png"), label) # Graficar Diversidad

        viz_seed = args.seed + 999_999 # Semilla aleatoria lejos para snapshot
        print(f"Corrida dedicada para distribución espacial de {problem.name} (semilla={viz_seed})")
        snapshots = capture_snapshots(problem, config, viz_seed) # Captura de snapshots
        plot_spatial_distribution( # Graficar distribución espacial de la snapshot
            problem, snapshots, os.path.join(args.output_dir, f"distribucion_espacial_{key}.png"), label
        )

    print_stats_table(stats_rows) # Impresión de estadísticas
    export_stats_csv(stats_rows, os.path.join(args.output_dir, "tabla_estadistica_pso.csv")) # Exportar estadísticas
    print("\nExperimento final completo. Resultados en:", os.path.abspath(args.output_dir))


"""
CLI / punto de entrada
____________________________________________________________
"""

# Pasar desde consola
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Actividad 5 - PSO: tuning y experimento final")
    p.add_argument("--mode", type=str, default="final", choices=["tuning", "final"])

    # Comunes
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iterations", type=int, default=300)
    p.add_argument("--topology", type=str, default="ring", choices=["ring", "global", "euclidean_knn"])
    p.add_argument("--neighborhood-k", type=int, default=1)
    p.add_argument("--inertia-schedule", type=str, default="constant", choices=["constant", "linear_decay"])
    p.add_argument("--w-start", type=float, default=0.9)
    p.add_argument("--w-end", type=float, default=0.4)
    p.add_argument("--boundary-handling", type=str, default="clip", choices=["clip", "reflect", "random_reset"])
    p.add_argument("--v-max-frac", type=float, default=0.2)
    p.add_argument("--stagnation-iterations", type=int, default=50)
    p.add_argument("--stagnation-epsilon", type=float, default=1e-10)
    p.add_argument("--diversity-epsilon", type=float, default=0.0)
    p.add_argument("--success-tolerance", type=float, default=1e-3)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)

    # Parámetros del enjambre (modo final; en tuning son la base)
    p.add_argument("--runs", type=int, default=N_RUNS_FINAL_DEFAULT)
    p.add_argument("--swarm-size", type=int, default=40)
    p.add_argument("--w", type=float, default=0.7)
    p.add_argument("--c1", type=float, default=1.5)
    p.add_argument("--c2", type=float, default=1.5)

    # Modo tuning
    p.add_argument("--tuning-runs", type=int, default=10)
    p.add_argument("--tuning-swarm-sizes", type=str, default="20,40,80")
    p.add_argument("--tuning-w", type=str, default="0.4,0.7,0.9")
    p.add_argument("--tuning-c1", type=str, default="1.0,1.5,2.0")
    p.add_argument("--tuning-c2", type=str, default="1.0,1.5,2.0")
    return p.parse_args()

# Función main
def main() -> None:
    args = parse_args()
    if args.mode == "tuning":
        rows = run_tuning(args)
        combined = combine_tuning_rows(rows)
        print_tuning_ranking(combined)
        os.makedirs(args.output_dir, exist_ok=True)
        export_tuning_csv(combined, os.path.join(args.output_dir, "tabla_sintonizacion_pso.csv"))
    else:
        run_final(args)


if __name__ == "__main__":
    main()
