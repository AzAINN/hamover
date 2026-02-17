"""Local embedded backend for validating multi-qubit bridge dynamics.

Despite the filename, this backend does not require QuTiP. It integrates the
full embedded Schrodinger equation directly with SciPy and returns leakage
and success diagnostics for the encoded subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from hamover.core.types import SU2Fields

from hamover.embedding import HamiltonianEmbedding


@dataclass(slots=True)
class EmbeddedSimulationResult:
    """Time history for a full embedded-state simulation."""

    t: np.ndarray
    states: np.ndarray
    probability_w: np.ndarray
    probability_r: np.ndarray
    leakage: np.ndarray
    max_leakage: float


def _pack_state(psi: np.ndarray) -> np.ndarray:
    return np.concatenate([np.real(psi), np.imag(psi)], axis=0)


def _unpack_state(y: np.ndarray, dim: int) -> np.ndarray:
    return y[:dim] + 1j * y[dim:]


def simulate_embedded(
    embedding: HamiltonianEmbedding,
    omega_func,
    Omega_func,
    x: float,
    T: float,
    n_points: int = 301,
    hbar: float = 1.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    leakage_warn: float = 1e-2,
) -> EmbeddedSimulationResult:
    """Simulate full n-qubit embedded dynamics and return subspace diagnostics."""
    if not (0.0 < x < 1.0):
        raise ValueError(f"x must be in (0,1), got {x}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if hbar <= 0.0:
        raise ValueError(f"hbar must be positive, got {hbar}")

    fields = SU2Fields(omega=omega_func, Omega=Omega_func, T=T)
    dim = embedding.dimension
    psi0 = embedding.initial_state(x)
    y0 = _pack_state(psi0)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        psi = _unpack_state(y, dim)
        H = embedding.embedded_hamiltonian(fields, float(t))
        psi_dot = -1j * (H @ psi) / hbar
        return _pack_state(psi_dot)

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, T),
        y0=y0,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"Embedded simulation failed: {sol.message}")

    t = np.linspace(0.0, T, n_points)
    y = sol.sol(t)
    states = y[:dim, :].T + 1j * y[dim:, :].T

    amp_w = np.einsum("j,nj->n", np.conj(embedding.w_encoded), states)
    amp_r = np.einsum("j,nj->n", np.conj(embedding.r_encoded), states)
    p_w = np.abs(amp_w) ** 2
    p_r = np.abs(amp_r) ** 2
    leakage = np.clip(1.0 - p_w - p_r, 0.0, None)
    max_leakage = float(np.max(leakage))

    if max_leakage > leakage_warn:
        warnings.warn(
            (
                "Embedded leakage exceeded threshold "
                f"({max_leakage:.3e} > {leakage_warn:.3e}); consider increasing penalty_strength"
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    return EmbeddedSimulationResult(
        t=t,
        states=states,
        probability_w=p_w,
        probability_r=p_r,
        leakage=leakage,
        max_leakage=max_leakage,
    )
