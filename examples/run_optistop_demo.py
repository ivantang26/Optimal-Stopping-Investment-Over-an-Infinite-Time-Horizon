"""End-to-end demo matching the PRD requirements."""

from __future__ import annotations

import os

import numpy as np

from optistop import (
    GBM,
    Grid,
    KinkedUtility,
    McDonaldSiegelAnalytical,
    PSORSolver,
    PrimalObstacleSolver,
    ResultFrame,
    VanillaCall,
    vega_from_triggers,
)
from optistop.solvers.fdm import FDMOperator
from optistop.viz.plots import plot_boundary_heatmap, plot_path_overlay, plot_value_vs_payoff


def run_smooth_benchmark(output_dir: str) -> float:
    r, mu, sigma, I = 0.05, 0.02, 0.2, 1.0
    process = GBM(mu=mu, sigma=sigma)
    utility = VanillaCall(investment_cost=I)
    grid = Grid(x_min=1e-4, x_max=8.0, n=2000)
    operator = FDMOperator(process=process, r=r)
    psor = PSORSolver(omega=1.2, tol=1e-7, max_iter=40_000)
    primal = PrimalObstacleSolver(grid=grid, utility=utility, operator=operator, psor=psor)
    result = primal.solve()

    analytical = McDonaldSiegelAnalytical(r=r, mu=mu, sigma=sigma, investment_cost=I)
    x_star_ref = analytical.trigger()
    rel_error = abs(result.trigger - x_star_ref) / x_star_ref
    print(f"[Benchmark] Numerical x*: {result.trigger:.6f}, analytical x*: {x_star_ref:.6f}, rel error: {rel_error:.4%}")

    plot_value_vs_payoff(result, os.path.join(output_dir, "value_vs_payoff.png"))
    ResultFrame(result).to_df().to_csv(os.path.join(output_dir, "value_table.csv"), index=False)
    return result.trigger


def run_kinked_case(output_dir: str) -> float:
    r, mu, sigma = 0.05, 0.015, 0.3
    process = GBM(mu=mu, sigma=sigma)
    utility = KinkedUtility(thresholds=(1.2, 2.0), slopes=(0.0, 0.2, 0.08), intercept=0.0)
    grid = Grid(x_min=1e-4, x_max=6.0, n=2000)
    operator = FDMOperator(process=process, r=r)
    psor = PSORSolver(omega=1.25, tol=1e-6, max_iter=40_000)
    result = PrimalObstacleSolver(grid=grid, utility=utility, operator=operator, psor=psor).solve()
    print(f"[Kinked utility] Trigger x*: {result.trigger:.6f} | converged={result.converged} iter={result.iterations}")
    plot_value_vs_payoff(result, os.path.join(output_dir, "kinked_value_vs_payoff.png"))
    return result.trigger


def run_sensitivity(output_dir: str) -> None:
    r = 0.05
    mu_grid = np.linspace(0.01, 0.08, 8)
    sigma_grid = np.linspace(0.1, 0.5, 10)
    trigger_surface = np.zeros((sigma_grid.size, mu_grid.size), dtype=np.float64)
    utility = VanillaCall(investment_cost=1.0)
    grid = Grid(x_min=1e-4, x_max=8.0, n=1000)
    psor = PSORSolver(omega=1.2, tol=1e-6, max_iter=20_000)

    for i, sigma in enumerate(sigma_grid):
        for j, mu in enumerate(mu_grid):
            process = GBM(mu=mu, sigma=sigma)
            operator = FDMOperator(process=process, r=r)
            result = PrimalObstacleSolver(grid=grid, utility=utility, operator=operator, psor=psor).solve()
            trigger_surface[i, j] = result.trigger

    # Report Vega across volatility at central drift.
    mid = mu_grid.size // 2
    vegas = vega_from_triggers(sigma_grid, trigger_surface[:, mid])
    print(f"[Sensitivity] Median Vega wrt sigma: {float(np.median(vegas)):.4f}")
    plot_boundary_heatmap(mu_grid, sigma_grid, trigger_surface, os.path.join(output_dir, "boundary_heatmap.html"))


def run_path_overlay(output_dir: str, trigger: float) -> None:
    process = GBM(mu=0.03, sigma=0.25)
    T, dt, n_paths = 3.0, 0.01, 25
    paths = process.simulate_paths(T=T, dt=dt, n_paths=n_paths, x0=1.0)
    n_steps = paths.shape[1] - 1
    time_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)
    plot_path_overlay(time_grid, paths, trigger, os.path.join(output_dir, "path_overlay.png"))


if __name__ == "__main__":
    out = "output/optistop"
    os.makedirs(out, exist_ok=True)
    smooth_trigger = run_smooth_benchmark(out)
    _ = run_kinked_case(out)
    run_sensitivity(out)
    run_path_overlay(out, smooth_trigger)
    print(f"Artifacts saved to: {out}")

