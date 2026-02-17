# hamover

`hamover` is a continuous-time quantum search framework based on Hamiltonian evolution. 
It supports Grover-like O(sqrt(N)) Scaling and Fixed Point Convergence.

It provides one orchestration API (`HamoverSearch`) across theory-first simulation, embedding
validation, and backend-oriented execution paths.

## What It Includes

- Multiple search Hamiltonian families: `farhi_gutmann`, `adiabatic`, `nonadiabatic`, `monotonic`, `custom_su2`
- Schedule support for adiabatic/nonadiabatic workflows
- Unified execution backends: `none`, `qutip`, `sim_annealing`, `classiq`, `ionq`, `quera`
- Embedding and bridge components for hardware-aligned workflows
- A single public entry point: `HamoverSearch(...).setup(...).solve()`

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[simulation]"    # QuTiP path
pip install -e ".[classiq]"       # Classiq pathway
pip install -e ".[hardware]"      # SimuQ + D-Wave stack
pip install -e ".[all]"           # Everything
```

## Quick Start

```python
from hamover import HamoverSearch

result = HamoverSearch(N=1024, target=42).setup(
    approach="adiabatic", #nonadiabatic ...
    backend="none",
    epsilon=0.5,
).solve()

print(result.target_found, result.probability, result.runtime)
```

## Paper References

1. *Hamiltonian embedding and hamming encoding for solving optimization problems on quantum hardware* (arXiv:2401.08550).  
   https://arxiv.org/abs/2401.08550
2. *Continuous-time quantum search and time-dependent two-level quantum systems* (arXiv:1903.11188).  
   https://arxiv.org/abs/1903.11188
