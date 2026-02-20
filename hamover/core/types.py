"""Core datatypes shared across the hamover package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


@dataclass(slots=True)
class BasisReduction:
    """Result of Gram-Schmidt reduction of {|w>, |s>} to {|w>, |r>}."""

    x: float
    w_state: np.ndarray
    r_state: np.ndarray
    s_state: np.ndarray
    N: int

    def __post_init__(self) -> None:
        if not (0.0 < float(self.x) < 1.0):
            raise ValueError(f"x must be in (0, 1), got {self.x}")
        if self.N < 2:
            raise ValueError(f"N must be >= 2, got {self.N}")


@dataclass(slots=True)
class HamiltonianMatrix:
    """2x2 matrix elements in {|w>, |r>} basis."""

    h11: complex
    h12: complex
    h21: complex
    h22: complex

    def to_array(self) -> np.ndarray:
        return np.array([[self.h11, self.h12], [self.h21, self.h22]], dtype=complex)

    def is_hermitian(self, atol: float = 1e-12) -> bool:
        matrix = self.to_array()
        return np.allclose(matrix, matrix.conj().T, atol=atol, rtol=0.0)


@dataclass(slots=True)
class SU2Fields:
    """Control fields in su(2) parametrization."""

    omega: Callable[[float], complex]
    Omega: Callable[[float], float]
    T: Optional[float]


@dataclass(slots=True)
class TransitionResult:
    """Result of computing transition probability."""

    P_sw: Callable[[float], float]
    P_sr: Callable[[float], float]
    t_star: Optional[float]
    P_max: Optional[float]


@dataclass(slots=True)
class EmbeddingSpec:
    """Specification of a Hamiltonian embedding."""

    n_qubits: int
    subspace_basis: list[np.ndarray]
    pauli_terms: list[tuple[str, Callable[[float], complex]]]
    penalty_strength: float
    scaling_factor: float


@dataclass(slots=True)
class BackendConfig:
    """Configuration for a quantum backend."""

    name: str
    shots: int
    api_token: Optional[str] = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchProblem:
    """Unstructured search problem definition."""

    N: int
    target_index: int
    items: Optional[Sequence[Any]] = None

    def __post_init__(self) -> None:
        if self.N < 2:
            raise ValueError(f"N must be >= 2, got {self.N}")
        if not (0 <= self.target_index < self.N):
            raise ValueError(f"target_index must lie in [0, {self.N - 1}]")
        if self.items is not None and len(self.items) != self.N:
            raise ValueError(f"items length must equal N={self.N}, got {len(self.items)}")


@dataclass(slots=True)
class SearchSetup:
    """Runtime setup options for the HamoverSearch entry point."""

    approach: str = "adiabatic"
    backend: str = "none"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    """End-to-end search outcome exposed by the top-level API."""

    target_found: bool
    probability: float
    runtime: float
    scaling_exponent: Optional[float]
    target_index: int
    selected_index: Optional[int]
    selected_item: Optional[Any]
    overlap_x: float
    approach: str
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BackendConfig",
    "BasisReduction",
    "EmbeddingSpec",
    "HamiltonianMatrix",
    "SU2Fields",
    "SearchProblem",
    "SearchResult",
    "SearchSetup",
    "TransitionResult",
]
