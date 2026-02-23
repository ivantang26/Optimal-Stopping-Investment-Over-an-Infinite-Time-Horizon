# OptiStop

Institutional-style Python engine for infinite-horizon optimal stopping with non-smooth utility and free-boundary methods.

## What is implemented

- `optistop.core`: abstract interfaces and solver result container.
- `optistop.processes`: `GBM`, `OrnsteinUhlenbeck`, `JumpDiffusion`.
- `optistop.payoffs`: `VanillaCall`, `PowerUtility`, `KinkedUtility`, convex conjugate helper.
- `optistop.solvers`: finite-difference operator, PSOR obstacle solver, dual transformation solver, analytical McDonald-Siegel benchmark.
- `optistop.analytics`: trigger/Greeks/DataFrame export utilities.
- `optistop.viz`: value/payoff plot, path-overlay plot, trigger sensitivity heatmap.

## Install

```bash
pip install -e .[viz,dev]
```

## Run demo

```bash
python examples/run_optistop_demo.py
```

Artifacts are generated under `output/optistop/`:

- `value_vs_payoff.png`
- `kinked_value_vs_payoff.png`
- `path_overlay.png`
- `boundary_heatmap.html`
- `value_table.csv`

## Run tests

```bash
pytest -q
```

## Notes

- Infinite-horizon setup enforces `r > 0`.
- Sparse matrices (`scipy.sparse`) are used in the PDE operator.
- PSOR uses `numba` acceleration when available.
