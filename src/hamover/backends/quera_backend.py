"""QuEra-oriented bridge backend utilities.

This module maps su(2) controls to Rydberg-style global controls
(Rabi frequency and detuning) and keeps a local simulation path for validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hamover.embedding import HamiltonianEmbedding
from .qutip_backend import EmbeddedSimulationResult, simulate_embedded


@dataclass(slots=True)
class QuEraProgram:
    """Control traces for a QuEra/Bloqade-style execution."""

    t: np.ndarray
    rabi_frequency: np.ndarray
    detuning: np.ndarray
    blockade_strength: float


def compile_quera_program(
    embedding: HamiltonianEmbedding,
    omega_func,
    Omega_func,
    T: float,
    n_points: int = 200,
) -> QuEraProgram:
    """Build global-control traces from su(2) fields.

    Mapping convention used here:
    - |omega(t)| -> Rabi frequency
    - Omega(t)   -> negative detuning
    - penalty    -> effective blockade proxy
    """
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    t = np.linspace(0.0, T, n_points)
    rabi = np.array([2.0 * abs(complex(omega_func(float(tt)))) for tt in t], dtype=float)
    detuning = np.array([-2.0 * float(Omega_func(float(tt))) for tt in t], dtype=float)

    return QuEraProgram(
        t=t,
        rabi_frequency=rabi,
        detuning=detuning,
        blockade_strength=float(embedding.penalty_strength),
    )


def simulate_quera_locally(
    embedding: HamiltonianEmbedding,
    omega_func,
    Omega_func,
    x: float,
    T: float,
    n_points: int = 301,
) -> EmbeddedSimulationResult:
    """Use the local embedded solver as a verification proxy for QuEra runs."""
    return simulate_embedded(
        embedding=embedding,
        omega_func=omega_func,
        Omega_func=Omega_func,
        x=x,
        T=T,
        n_points=n_points,
    )
