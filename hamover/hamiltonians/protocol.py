"""Protocol definitions for common two-level Hamiltonian interfaces."""

from typing import Protocol, runtime_checkable

import numpy as np

from hamover.core.types import SU2Fields


@runtime_checkable
class TwoLevelHamiltonian(Protocol):
    """Common interface for 2x2 Hamiltonians."""

    def matrix(self, t: float) -> np.ndarray:
        """Return 2x2 Hamiltonian matrix at time t."""

    def to_su2(self) -> SU2Fields:
        """Convert to (omega(t), Omega(t)) parametrization."""

    def energy_gap(self, t: float) -> float:
        """Return instantaneous gap lambda_plus(t) - lambda_minus(t)."""

    @property
    def is_time_dependent(self) -> bool:
        """Whether matrix(t) depends on t."""


SearchHamiltonian = TwoLevelHamiltonian

__all__ = ["SearchHamiltonian", "TwoLevelHamiltonian"]
