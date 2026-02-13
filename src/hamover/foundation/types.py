from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
