"""Bridge from embedded Hamiltonians to SimuQ-compatible piecewise programs.

This module converts a full embedded Hamiltonian H_embed(t) into piecewise-constant
Pauli segments, optionally materializing a real SimuQ ``QSystem`` when the ``simuq``
package is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from hamover.core.types import SU2Fields
from hamover.hamiltonians.controlled import SU2Hamiltonian
from hamover.solver import evolve

from hamover.embedding import HamiltonianEmbedding
from hamover.backends.qutip_backend import simulate_embedded


@dataclass(slots=True)
class SimuQSegment:
    """One piecewise-constant evolution segment in Pauli form."""

    t_start: float
    t_end: float
    duration: float
    pauli_terms: list[tuple[str, complex]]


@dataclass(slots=True)
class BuiltSimuQSystem:
    """Portable representation of a compiled SimuQ program.

    ``qsystem`` is populated only when the optional ``simuq`` dependency is present.
    """

    n_qubits: int
    total_time: float
    n_segments: int
    segments: list[SimuQSegment]
    qsystem: Optional[Any]


@dataclass(slots=True)
class SimuQValidation:
    """Numerical agreement report between reference and embedded pipelines."""

    p_reference_final: float
    p_embedded_final: float
    p_simuq_final: Optional[float]
    error_embedded: float
    error_simuq: Optional[float]
    max_leakage: float


def _as_real_coeff(coeff: complex, tol: float = 1e-10) -> float:
    if abs(np.imag(coeff)) > tol:
        raise ValueError(f"Pauli coefficient has non-negligible imaginary part: {coeff}")
    return float(np.real(coeff))


def _pauli_operator_from_string(qubits: list[Any], pauli_string: str) -> Any:
    if pauli_string == "I":
        return None

    out = None
    for token in pauli_string.split("*"):
        axis = token[0]
        qidx = int(token[1:])
        term = getattr(qubits[qidx], axis)
        out = term if out is None else out * term
    return out


def _build_qsystem_if_available(segments: list[SimuQSegment], n_qubits: int) -> Optional[Any]:
    try:
        from simuq import QSystem, Qubit
    except Exception:
        return None

    qs = QSystem()
    qubits = [Qubit(qs, name=f"Q{i}") for i in range(n_qubits)]

    for seg in segments:
        h_expr = 0
        for pauli_string, coeff in seg.pauli_terms:
            op = _pauli_operator_from_string(qubits, pauli_string)
            if op is None:
                continue
            h_expr = h_expr + _as_real_coeff(coeff) * op
        qs.add_evolution(h_expr, seg.duration)

    return qs


def build_simuq_system(
    embedding: HamiltonianEmbedding,
    omega_func,
    Omega_func,
    T: float,
    n_segments: int = 100,
    tol: float = 1e-10,
) -> BuiltSimuQSystem:
    """Compile embedded controls into piecewise Pauli segments.

    The resulting object is backend-agnostic. If ``simuq`` is installed, it also
    includes a concrete ``QSystem`` ready for provider compilation.
    """
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    fields = SU2Fields(omega=omega_func, Omega=Omega_func, T=T)
    dt = float(T) / float(n_segments)

    segments: list[SimuQSegment] = []
    for k in range(n_segments):
        t_start = k * dt
        t_end = (k + 1) * dt
        t_mid = 0.5 * (t_start + t_end)
        coeffs = embedding.pauli_coefficients_at(fields, t_mid, tol=tol)
        segments.append(
            SimuQSegment(
                t_start=t_start,
                t_end=t_end,
                duration=dt,
                pauli_terms=coeffs,
            )
        )

    qs = _build_qsystem_if_available(segments, embedding.n_qubits)
    return BuiltSimuQSystem(
        n_qubits=embedding.n_qubits,
        total_time=float(T),
        n_segments=n_segments,
        segments=segments,
        qsystem=qs,
    )


def _try_run_simuq_qutip(
    built: BuiltSimuQSystem,
    embedding: HamiltonianEmbedding,
) -> Optional[float]:
    if built.qsystem is None:
        return None

    try:
        from simuq.qutip import QuTiPProvider
    except Exception:
        return None

    provider = QuTiPProvider()
    provider.compile(built.qsystem)
    provider.run()
    probs = provider.results()

    p_w = 0.0
    for bitstring, prob in probs.items():
        idx = int(bitstring, 2)
        p_w += float(prob) * float(abs(embedding.w_encoded[idx]) ** 2)
    return p_w


def validate_simuq_system(
    built: BuiltSimuQSystem,
    embedding: HamiltonianEmbedding,
    omega_func,
    Omega_func,
    x: float,
    T: float,
    n_points: int = 201,
) -> SimuQValidation:
    """Validate end-to-end bridge consistency against 2x2 reference dynamics."""
    if not (0.0 < x < 1.0):
        raise ValueError(f"x must be in (0,1), got {x}")

    su2 = SU2Hamiltonian(omega_func=omega_func, Omega_func=Omega_func, T=T)
    ref = evolve(su2, x=x, T=T, n_points=n_points)
    p_ref = float(ref.probability_sw[-1])

    embedded = simulate_embedded(
        embedding=embedding,
        omega_func=omega_func,
        Omega_func=Omega_func,
        x=x,
        T=T,
        n_points=n_points,
    )
    p_emb = float(embedded.probability_w[-1])

    p_simuq = _try_run_simuq_qutip(built, embedding)
    err_emb = abs(p_emb - p_ref)
    err_simuq = abs(p_simuq - p_ref) if p_simuq is not None else None

    return SimuQValidation(
        p_reference_final=p_ref,
        p_embedded_final=p_emb,
        p_simuq_final=p_simuq,
        error_embedded=err_emb,
        error_simuq=err_simuq,
        max_leakage=float(np.max(embedded.leakage)),
    )
