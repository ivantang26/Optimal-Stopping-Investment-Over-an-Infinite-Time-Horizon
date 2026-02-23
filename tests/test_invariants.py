"""Core mathematical invariants."""

from __future__ import annotations

import numpy as np

from optistop import GBM, Grid, PSORSolver, PrimalObstacleSolver, VanillaCall
from optistop.solvers.fdm import FDMOperator


def solve_one(mu: float, sigma: float):
    process = GBM(mu=mu, sigma=sigma)
    utility = VanillaCall(investment_cost=1.0)
    grid = Grid(x_min=1e-4, x_max=10.0, n=1500)
    operator = FDMOperator(process=process, r=0.05)
    psor = PSORSolver(omega=1.2, tol=1e-6, max_iter=30_000)
    return PrimalObstacleSolver(grid=grid, utility=utility, operator=operator, psor=psor).solve()


def test_obstacle_condition_and_positivity() -> None:
    rng = np.random.default_rng(123)
    for _ in range(5):
        mu = float(rng.uniform(0.01, 0.08))
        sigma = float(rng.uniform(0.1, 0.4))
        result = solve_one(mu, sigma)
        assert np.all(result.value >= result.payoff - 1e-8)
        assert np.all(result.value >= -1e-8)


def test_trigger_monotone_with_sigma() -> None:
    sigmas = np.array([0.12, 0.18, 0.24, 0.30, 0.36], dtype=np.float64)
    triggers = []
    for sigma in sigmas:
        triggers.append(solve_one(mu=0.03, sigma=float(sigma)).trigger)
    t = np.array(triggers)
    assert np.all(np.diff(t) >= -1e-3)

