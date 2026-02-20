"""Basis-reduction and overlap helpers for the two-dimensional search subspace."""

import numpy as np

from .types import BasisReduction


def _validate_normalized(state: np.ndarray, name: str, atol: float = 1e-12) -> None:
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"{name} must be normalized, ||{name}||={norm}")


def reduce_to_two_level(w_state: np.ndarray, s_state: np.ndarray) -> BasisReduction:
    """Gram-Schmidt reduction {|w>, |s>} -> {|w>, |r>} (paper Eqs. 4-9)."""
    w = np.asarray(w_state, dtype=complex).flatten()
    s = np.asarray(s_state, dtype=complex).flatten()

    if w.shape != s.shape:
        raise ValueError(f"State dimensions must match: {w.shape} vs {s.shape}")
    if w.size < 2:
        raise ValueError("State dimension must be at least 2")

    _validate_normalized(w, "w_state")
    _validate_normalized(s, "s_state")

    x_complex = np.vdot(w, s)
    s_aligned = s

    # The paper uses real x = <w|s>; remove global phase from |s> if needed.
    if abs(x_complex.imag) > 1e-10:
        phase = np.exp(-1j * np.angle(x_complex))
        s_aligned = s * phase
        x_complex = np.vdot(w, s_aligned)

    x = float(np.real(x_complex))
    if not (0.0 < x < 1.0):
        raise ValueError(f"Overlap x must be in (0, 1), got {x}")

    r = (s_aligned - x * w) / np.sqrt(1.0 - x**2)

    ortho = np.vdot(w, r)
    if not np.isclose(ortho, 0.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"Orthogonality check failed, <w|r>={ortho}")
    _validate_normalized(r, "r_state")

    return BasisReduction(x=x, w_state=w, r_state=r, s_state=s_aligned, N=w.size)


def uniform_source_state(N: int) -> np.ndarray:
    """Return the uniform superposition |s> = (1/sqrt(N)) sum_i |i>."""
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    return np.ones(N, dtype=complex) / np.sqrt(N)


def standard_target_state(N: int, target_index: int = 0) -> np.ndarray:
    """Return basis target state |w> = |target_index>."""
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    if not (0 <= target_index < N):
        raise ValueError(f"target_index must be in [0, {N-1}], got {target_index}")

    w = np.zeros(N, dtype=complex)
    w[target_index] = 1.0
    return w


def reduced_initial_state(x: float) -> np.ndarray:
    """Return |s> represented in {|w>, |r>} as [x, sqrt(1-x^2)]^T."""
    if not (0.0 < x < 1.0):
        raise ValueError(f"x must be in (0, 1), got {x}")
    return np.array([x, np.sqrt(1.0 - x**2)], dtype=complex)
