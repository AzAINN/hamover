"""Unified solver: closed-form, Schrödinger ODE, and propagator evolution methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from hamover.core.basis import reduced_initial_state
from hamover.core.types import TransitionResult


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class StaticHamiltonian(Protocol):
    hbar: float

    def matrix(self, t: float = 0.0) -> np.ndarray:
        ...


class MatrixHamiltonian(Protocol):
    def matrix(self, t: float) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Closed-form solver
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ClosedFormSolution:
    """Analytical transition-probability trajectory."""

    t: np.ndarray
    probability_sw: np.ndarray
    probability_sr: np.ndarray
    transition: TransitionResult


def _build_transition_closed(t: np.ndarray, p_sw: np.ndarray, t_star: float | None, p_max: float | None) -> TransitionResult:
    p_sr = 1.0 - p_sw

    def P_sw_eval(tq: float) -> float:
        return float(np.interp(tq, t, p_sw, left=p_sw[0], right=p_sw[-1]))

    def P_sr_eval(tq: float) -> float:
        return float(np.interp(tq, t, p_sr, left=p_sr[0], right=p_sr[-1]))

    return TransitionResult(P_sw=P_sw_eval, P_sr=P_sr_eval, t_star=t_star, P_max=p_max)


def solve_static(hamiltonian: StaticHamiltonian, t_array: np.ndarray, x: float | None = None) -> ClosedFormSolution:
    """Closed-form solver for time-independent 2x2 Hamiltonians (GQS/FG style)."""
    t = np.asarray(t_array, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t_array must be 1D with at least two points")

    x_val = float(x if x is not None else getattr(hamiltonian, "x"))
    if not (0.0 < x_val < 1.0):
        raise ValueError(f"x must be in (0,1), got {x_val}")

    H = np.asarray(hamiltonian.matrix(0.0), dtype=complex)
    if H.shape != (2, 2):
        raise ValueError(f"Static Hamiltonian must be 2x2, got {H.shape}")

    h11, h12, h21, h22 = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    hbar = float(getattr(hamiltonian, "hbar", 1.0))

    denom = h12 * h21 / hbar**2 + (h11 - h22) ** 2 / (4.0 * hbar**2)
    if abs(denom) < 1e-16:
        raise ValueError("Degenerate closed-form denominator for static solver")

    Omega = np.sqrt(denom)
    A = (
        0.5 * (h11 - h22) / hbar * x_val + h12 / hbar * np.sqrt(1.0 - x_val**2)
    ) ** 2 / denom

    p_sw = np.real(x_val**2 * np.cos(Omega * t) ** 2 + A * np.sin(Omega * t) ** 2)
    p_sw = np.clip(p_sw, 0.0, 1.0)
    p_sr = 1.0 - p_sw

    if hasattr(hamiltonian, "t_star"):
        t_star = float(hamiltonian.t_star())
    else:
        vals = np.linalg.eigvalsh(H)
        gap = float(vals[1] - vals[0])
        t_star = float(np.pi * hbar / gap)

    p_max = float(getattr(hamiltonian, "P_max", lambda: np.max(p_sw))())

    transition = _build_transition_closed(t, p_sw, t_star=t_star, p_max=p_max)
    return ClosedFormSolution(t=t, probability_sw=p_sw, probability_sr=p_sr, transition=transition)


def solve_rabi(hamiltonian: object, t_array: np.ndarray, x: float | None = None) -> ClosedFormSolution:
    """Closed-form solver for Rabi Hamiltonian (paper Eqs. 24-25)."""
    t = np.asarray(t_array, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t_array must be 1D with at least two points")

    x_val = float(x if x is not None else getattr(hamiltonian, "x"))
    if not (0.0 <= x_val <= 1.0):
        raise ValueError(f"x must be in [0,1], got {x_val}")

    Gamma = float(getattr(hamiltonian, "Gamma"))
    omega = float(getattr(hamiltonian, "omega"))
    omega_21 = float(getattr(hamiltonian, "omega_21"))
    hbar = float(getattr(hamiltonian, "hbar", 1.0))

    delta = omega - omega_21
    Omega = np.sqrt(Gamma**2 / hbar**2 + delta**2 / 4.0)
    D = (delta / 2.0 * x_val - Gamma / hbar * np.sqrt(1.0 - x_val**2)) / Omega

    p_sw = np.real(x_val**2 * np.cos(Omega * t) ** 2 + D**2 * np.sin(Omega * t) ** 2)
    p_sw = np.clip(p_sw, 0.0, 1.0)
    p_sr = 1.0 - p_sw

    transition = _build_transition_closed(t, p_sw, t_star=None, p_max=float(np.max(p_sw)))
    return ClosedFormSolution(t=t, probability_sw=p_sw, probability_sr=p_sr, transition=transition)


def solve_monotonic(model: object, t_array: np.ndarray, x: float) -> ClosedFormSolution:
    """Closed-form-like evaluation for monotonic scenarios using analytical alpha/beta laws."""
    t = np.asarray(t_array, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t_array must be 1D with at least two points")
    if not (0.0 < x < 1.0):
        raise ValueError(f"x must be in (0,1), got {x}")

    if not hasattr(model, "transition_probability"):
        raise ValueError("Model must provide transition_probability(t,x)")

    p_sw = np.array([float(model.transition_probability(float(tt), x)) for tt in t], dtype=float)
    p_sw = np.clip(p_sw, 0.0, 1.0)
    p_sr = 1.0 - p_sw

    transition = _build_transition_closed(t, p_sw, t_star=None, p_max=float(np.max(p_sw)))
    return ClosedFormSolution(t=t, probability_sw=p_sw, probability_sr=p_sr, transition=transition)


# ---------------------------------------------------------------------------
# Schrödinger ODE solver
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SchrodingerSolution:
    """Numerical evolution result for i*hbar*dpsi/dt = H(t)psi."""

    t: np.ndarray
    psi: np.ndarray
    probability_sw: np.ndarray
    probability_sr: np.ndarray
    transition: TransitionResult
    norm_deviation_max: float


def _build_transition_sch(t: np.ndarray, p_sw: np.ndarray) -> TransitionResult:
    p_sr = 1.0 - p_sw

    def P_sw_eval(tq: float) -> float:
        return float(np.interp(tq, t, p_sw, left=p_sw[0], right=p_sw[-1]))

    def P_sr_eval(tq: float) -> float:
        return float(np.interp(tq, t, p_sr, left=p_sr[0], right=p_sr[-1]))

    return TransitionResult(P_sw=P_sw_eval, P_sr=P_sr_eval, t_star=None, P_max=None)


def _rhs_real(t: float, y: np.ndarray, hamiltonian: MatrixHamiltonian, hbar: float) -> np.ndarray:
    psi = y[:2] + 1j * y[2:]
    Hpsi = hamiltonian.matrix(t) @ psi
    psi_dot = -1j * Hpsi / hbar
    return np.array([psi_dot[0].real, psi_dot[1].real, psi_dot[0].imag, psi_dot[1].imag], dtype=float)


def evolve(
    hamiltonian: MatrixHamiltonian,
    x: float,
    T: float,
    n_points: int,
    hbar: float = 1.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "RK45",
) -> SchrodingerSolution:
    """General Schrödinger evolution for any 2x2 Hamiltonian with matrix(t)."""
    if not (0.0 < x < 1.0):
        raise ValueError(f"x must be in (0,1), got {x}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if hbar <= 0.0:
        raise ValueError(f"hbar must be positive, got {hbar}")

    psi0 = reduced_initial_state(x)
    y0 = np.array([psi0[0].real, psi0[1].real, psi0[0].imag, psi0[1].imag], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: _rhs_real(t, y, hamiltonian, hbar),
        t_span=(0.0, T),
        y0=y0,
        method=method,
        dense_output=True,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Schrodinger integration failed: {sol.message}")

    t_grid = np.linspace(0.0, T, n_points)
    y_grid = sol.sol(t_grid)
    psi = y_grid[:2, :].T + 1j * y_grid[2:, :].T

    p_sw = np.abs(psi[:, 0]) ** 2
    p_sr = np.abs(psi[:, 1]) ** 2
    norms = p_sw + p_sr
    norm_dev = float(np.max(np.abs(norms - 1.0)))

    transition = _build_transition_sch(t_grid, p_sw)
    return SchrodingerSolution(
        t=t_grid,
        psi=psi,
        probability_sw=p_sw,
        probability_sr=p_sr,
        transition=transition,
        norm_deviation_max=norm_dev,
    )


def evolve_interaction(
    H0: np.ndarray,
    V_func: Callable[[float], np.ndarray],
    x: float,
    T: float,
    n_points: int,
    hbar: float = 1.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "RK45",
) -> SchrodingerSolution:
    """Integrate in the interaction picture for H(t)=H0+V(t)."""
    H0 = np.asarray(H0, dtype=complex)
    if H0.shape != (2, 2):
        raise ValueError(f"H0 must be 2x2, got {H0.shape}")

    class _InteractionHamiltonian:
        def matrix(self, t: float) -> np.ndarray:
            U0 = expm(-1j * H0 * t / hbar)
            U0_dag = U0.conj().T
            Vt = np.asarray(V_func(t), dtype=complex)
            return U0_dag @ Vt @ U0

    sol_I = evolve(
        hamiltonian=_InteractionHamiltonian(),
        x=x,
        T=T,
        n_points=n_points,
        hbar=hbar,
        rtol=rtol,
        atol=atol,
        method=method,
    )

    psi_s = np.empty_like(sol_I.psi)
    for i, t in enumerate(sol_I.t):
        U0 = expm(-1j * H0 * t / hbar)
        psi_s[i] = U0 @ sol_I.psi[i]

    p_sw = np.abs(psi_s[:, 0]) ** 2
    p_sr = np.abs(psi_s[:, 1]) ** 2
    norms = p_sw + p_sr
    norm_dev = float(np.max(np.abs(norms - 1.0)))

    transition = _build_transition_sch(sol_I.t, p_sw)
    return SchrodingerSolution(
        t=sol_I.t,
        psi=psi_s,
        probability_sw=p_sw,
        probability_sr=p_sr,
        transition=transition,
        norm_deviation_max=norm_dev,
    )


def sample_at(solution: SchrodingerSolution, t_query: float) -> tuple[complex, complex, float]:
    """Interpolate psi0, psi1, and P_sw at a query time."""
    re0 = np.interp(t_query, solution.t, solution.psi[:, 0].real)
    im0 = np.interp(t_query, solution.t, solution.psi[:, 0].imag)
    re1 = np.interp(t_query, solution.t, solution.psi[:, 1].real)
    im1 = np.interp(t_query, solution.t, solution.psi[:, 1].imag)
    p = solution.transition.P_sw(t_query)
    return complex(re0, im0), complex(re1, im1), p


# ---------------------------------------------------------------------------
# Propagator solver
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PropagatorSolution:
    """Stepwise unitary propagation result."""

    t: np.ndarray
    U: np.ndarray
    psi: np.ndarray
    probability_sw: np.ndarray
    probability_sr: np.ndarray
    transition: TransitionResult
    unitarity_error_max: float


def propagator_step(H_mid: np.ndarray, dt: float, hbar: float = 1.0) -> np.ndarray:
    """Compute U(dt)=exp(-i H_mid dt / hbar)."""
    if hbar <= 0.0:
        raise ValueError(f"hbar must be positive, got {hbar}")
    return expm(-1j * np.asarray(H_mid, dtype=complex) * dt / hbar)


def compute_propagator(
    hamiltonian: MatrixHamiltonian,
    T: float,
    n_steps: int,
    hbar: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build unitary history U_k for times t_k via midpoint rule."""
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    U = np.eye(2, dtype=complex)
    U_hist = np.empty((n_steps + 1, 2, 2), dtype=complex)
    U_hist[0] = U

    max_err = 0.0
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        H_mid = hamiltonian.matrix(t_mid)
        U_step = propagator_step(H_mid, dt, hbar=hbar)
        U = U_step @ U
        U_hist[k + 1] = U

        err = np.linalg.norm(U.conj().T @ U - np.eye(2, dtype=complex))
        max_err = max(max_err, float(err))

    return t_grid, U_hist, max_err


def evolve_with_propagator(
    hamiltonian: MatrixHamiltonian,
    x: float,
    T: float,
    n_steps: int,
    hbar: float = 1.0,
) -> PropagatorSolution:
    """Evolve initial |s>=[x,sqrt(1-x^2)]^T using accumulated propagators."""
    if not (0.0 < x < 1.0):
        raise ValueError(f"x must be in (0,1), got {x}")

    t_grid, U_hist, unitary_err = compute_propagator(hamiltonian, T=T, n_steps=n_steps, hbar=hbar)
    psi0 = reduced_initial_state(x)

    psi_hist = np.einsum("kij,j->ki", U_hist, psi0)
    p_sw = np.abs(psi_hist[:, 0]) ** 2
    p_sr = np.abs(psi_hist[:, 1]) ** 2

    transition = _build_transition_sch(t_grid, p_sw)
    return PropagatorSolution(
        t=t_grid,
        U=U_hist,
        psi=psi_hist,
        probability_sw=p_sw,
        probability_sr=p_sr,
        transition=transition,
        unitarity_error_max=unitary_err,
    )


def convergence_delta(
    hamiltonian: MatrixHamiltonian,
    x: float,
    T: float,
    n_steps: int,
    hbar: float = 1.0,
) -> float:
    """Return |P_n - P_2n| at final time as a simple convergence diagnostic."""
    coarse = evolve_with_propagator(hamiltonian, x=x, T=T, n_steps=n_steps, hbar=hbar)
    fine = evolve_with_propagator(hamiltonian, x=x, T=T, n_steps=2 * n_steps, hbar=hbar)
    return float(abs(coarse.probability_sw[-1] - fine.probability_sw[-1]))


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SolverBundle:
    """Container returned by high-level solver helpers."""

    schrodinger: Optional[SchrodingerSolution] = None
    closed_form: Optional[ClosedFormSolution] = None
    propagator: Optional[PropagatorSolution] = None


def closed_form_gqs(hamiltonian: Any, t_array: np.ndarray, x: float | None = None) -> ClosedFormSolution:
    """Closed-form GQS/Farhi-Gutmann evaluation."""
    return solve_static(hamiltonian, t_array, x=x)


def numerical_solve(
    hamiltonian: Any,
    x: float,
    T: float,
    n_points: int = 401,
    with_propagator: bool = False,
    propagator_steps: int = 2000,
) -> SolverBundle:
    """Numerical Schrödinger evolution with optional propagator cross-check."""
    sch = evolve(hamiltonian, x=x, T=T, n_points=n_points)
    prop = None
    if with_propagator:
        prop = evolve_with_propagator(hamiltonian, x=x, T=T, n_steps=propagator_steps)
    return SolverBundle(schrodinger=sch, propagator=prop)


__all__ = [
    "ClosedFormSolution",
    "PropagatorSolution",
    "SchrodingerSolution",
    "SolverBundle",
    "closed_form_gqs",
    "compute_propagator",
    "convergence_delta",
    "evolve",
    "evolve_interaction",
    "evolve_with_propagator",
    "numerical_solve",
    "propagator_step",
    "sample_at",
    "solve_monotonic",
    "solve_rabi",
    "solve_static",
]
