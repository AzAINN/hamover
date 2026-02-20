"""Hardware-adapter layer for embedded Hamiltonian execution."""

from .dwave_backend import (
    AnnealScheduleTrace,
    DWaveAnnealResult,
    IsingModel,
    anneal_schedule_from_controls,
    extract_ising_model,
    run_simulated_annealing,
)
from .ionq_backend import IonQProgram, IonQSegment, compile_ionq_program, simulate_ionq_locally
from .qutip_backend import EmbeddedSimulationResult, simulate_embedded
from .quera_backend import QuEraProgram, compile_quera_program, simulate_quera_locally
from .classiq_backend import ClassiqResult, compile_classiq_circuit, run_classiq_search

__all__ = [
    "AnnealScheduleTrace",
    "ClassiqResult",
    "DWaveAnnealResult",
    "EmbeddedSimulationResult",
    "IonQProgram",
    "IonQSegment",
    "IsingModel",
    "QuEraProgram",
    "anneal_schedule_from_controls",
    "compile_classiq_circuit",
    "compile_ionq_program",
    "compile_quera_program",
    "extract_ising_model",
    "run_classiq_search",
    "run_simulated_annealing",
    "simulate_embedded",
    "simulate_ionq_locally",
    "simulate_quera_locally",
]
