# hamover

Grover-like analog (continuous-time) Hamiltonian search with a single entry point.

## Quick Start

```python
from hamover import HamoverSearch

search = HamoverSearch(N=1024, target=42)
result = search.setup(approach="adiabatic", backend="none", epsilon=0.5).solve()

print(result.target_found, result.probability, result.runtime)
```

## CLI Example

```bash
python3 experiments/run_unstructured_search.py --N 64 --target-index 7 --backend none
``` 
