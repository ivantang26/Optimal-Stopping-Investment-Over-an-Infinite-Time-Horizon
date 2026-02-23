"""Benchmark numerical trigger against McDonald-Siegel formula."""

from __future__ import annotations

from optistop import GBM, Grid, McDonaldSiegelAnalytical, PSORSolver, PrimalObstacleSolver, VanillaCall
from optistop.solvers.fdm import FDMOperator


def test_trigger_close_to_analytical() -> None:
    r, mu, sigma, I = 0.05, 0.02, 0.2, 1.0
    process = GBM(mu=mu, sigma=sigma)
    utility = VanillaCall(investment_cost=I)
    grid = Grid(x_min=1e-4, x_max=8.0, n=2500)
    operator = FDMOperator(process=process, r=r)
    psor = PSORSolver(omega=1.2, tol=1e-7, max_iter=50_000)
    result = PrimalObstacleSolver(grid=grid, utility=utility, operator=operator, psor=psor).solve()

    ref = McDonaldSiegelAnalytical(r=r, mu=mu, sigma=sigma, investment_cost=I).trigger()
    rel_error = abs(result.trigger - ref) / ref
    assert rel_error < 2.0e-2

