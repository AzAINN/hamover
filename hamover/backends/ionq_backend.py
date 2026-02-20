"""IonQ-oriented bridge backend utilities.

The real cloud execution path is intentionally lightweight here; this module
focuses on compiling su(2)-embedded controls into an IonQ-friendly segment
representation (XX + Z structure) and validating it via local simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hamover.embedding import HamiltonianEmbedding, pauli_string_to_label
from .qutip_backend import EmbeddedSimulationResult, simulate_embedded


@dataclass(slots=True)
class IonQSegment:
    """One time slice of an IonQ-oriented analog program."""

    duration: float
    xx_terms: dict[tuple[int, int], float]
    z_terms: dict[int, float]
    dropped_terms: int


@dataclass(slots=True)
class IonQProgram:
    """Piecewise analog program targeting IonQ-style interactions."""

    n_qubits: int
    total_time: float
    segments: list[IonQSegment]


def compile_ionq_program(
    embedding: HamiltonianEmbedding,
    omega_func,
    Omega_func,
    T: float,
    n_segments: int = 100,
    tol: float = 1e-10,
) -> IonQProgram:
    """Compile embedded Hamiltonians into XX+Z segment data."""
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    dt = float(T) / n_segments
    segments: list[IonQSegment] = []

    from hamover.core.types import SU2Fields

    fields = SU2Fields(omega=omega_func, Omega=Omega_func, T=T)
    for k in range(n_segments):
        t_mid = (k + 0.5) * dt
        coeffs = embedding.pauli_coefficients_at(fields, t_mid, tol=tol)

        xx_terms: dict[tuple[int, int], float] = {}
        z_terms: dict[int, float] = {}
        dropped = 0

        for pstring, coeff in coeffs:
            c = float(np.real(coeff))
            label = pauli_string_to_label(pstring, embedding.n_qubits)
            support = [i for i, axis in enumerate(label) if axis != "I"]
            axes = [label[i] for i in support]

            if len(support) == 1 and axes[0] == "Z":
                z_terms[support[0]] = z_terms.get(support[0], 0.0) + c
            elif len(support) == 2 and axes[0] == "X" and axes[1] == "X":
                i, j = support
                key = (i, j) if i < j else (j, i)
                xx_terms[key] = xx_terms.get(key, 0.0) + c
            elif len(support) == 0:
                continue
            else:
                dropped += 1

        segments.append(
            IonQSegment(
                duration=dt,
                xx_terms=xx_terms,
                z_terms=z_terms,
                dropped_terms=dropped,
            )
        )

    return IonQProgram(
        n_qubits=embedding.n_qubits,
        total_time=float(T),
        segments=segments,
    )


def simulate_ionq_locally(
    embedding: HamiltonianEmbedding,
    omega_func,
    Omega_func,
    x: float,
    T: float,
    n_points: int = 301,
) -> EmbeddedSimulationResult:
    """Use the local embedded solver as a verification proxy for IonQ runs."""
    return simulate_embedded(
        embedding=embedding,
        omega_func=omega_func,
        Omega_func=Omega_func,
        x=x,
        T=T,
        n_points=n_points,
    )
