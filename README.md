# hamover

`hamover` is a continuous-time quantum search library based on Hamiltonian evolution.

`[Note: Under Development]`

Use one API for all flows:

```python
HamoverSearch(...).setup(...).solve()
```

## Quick Setup

Requirements:
- Python `3.10+`

Install in editable mode:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[simulation]"  # QuTiP simulation backend
pip install -e ".[classiq]"     # Classiq backend
pip install -e ".[hardware]"    # SimuQ + D-Wave stack [Underdevelopment]
pip install -e ".[all]"         # everything
```

## 60-Second Example
Note: (schedule ="gap_power", p = 2.0) is the sweet spot for fixed-point behaviour and quadratic speedup
```python
from hamover import HamoverSearch

result = (
    HamoverSearch(N=1024, target=42)
    .setup(
        approach="adiabatic",
        backend="none",
        epsilon=0.5,
    )
    .solve()
)

print("found:", result.target_found)
print("probability:", result.probability)
print("runtime:", result.runtime)
print("diagnostics keys:", list(result.diagnostics.keys()))
```

## Main Options

Approaches:
- `farhi_gutmann`
- `adiabatic`
- `nonadiabatic`
- `monotonic`
- `custom_su2`

Backends:
- `none` (theory-only solver)
- `qutip`
- `sim_annealing`
- `classiq`
- `ionq`
- `quera`

## Typical Commands

Run tests:

```bash
pytest -q
```

Run notebooks:
- `hamoverSearch.ipynb`
- `comparison.ipynb`

## Troubleshooting

- `ModuleNotFoundError` for optional backend packages:
  install the matching extra, e.g. `pip install -e ".[classiq]"` or `pip install -e ".[simulation]"`.

- `HamoverSearch` import issues after pulling new changes:
  rerun `pip install -e .` in the repo root.

- Notebook behavior looks stale or inconsistent:
  restart the kernel, then run cells top-to-bottom once.

- Grover notebook path fails for non-power-of-two problem size:
  use `N = 2^n` (for example `4`, `8`, `16`, `32`).

- Backend not available in your environment:
  start with `backend="none"` to validate the model first, then switch to your target backend.

## References

1. Hamiltonian embedding and hamming encoding for solving optimization problems on quantum hardware  
   https://arxiv.org/abs/2401.08550
2. Continuous-time quantum search and time-dependent two-level quantum systems  
   https://arxiv.org/abs/1903.11188
