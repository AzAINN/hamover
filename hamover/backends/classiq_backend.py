"""Classiq qDRIFT gate-based execution path.

This backend compiles the embedded Hamiltonian to a gate-based circuit using
Classiq's qDRIFT (randomized product formula), then executes and decodes.
Unlike the D-Wave/IonQ backends, qDRIFT preserves *all* Pauli terms — no
term-dropping.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from hamover.core.types import SU2Fields
from hamover.embedding import HamiltonianEmbedding


@dataclass(slots=True)
class ClassiqResult:
    """Decoded measurement summary from Classiq qDRIFT execution."""

    success_probability: float
    found_count: int
    not_found_count: int
    leakage_count: int
    bitstrings: list[str]
    circuit_depth: int
    cx_count: int
    n_qubits: int
    num_qdrift: int


def _hamover_pauli_to_classiq(
    pauli_terms: list[tuple[str, complex]],
    n_qubits: int,
    imag_tol: float = 1e-8,
):
    """Convert hamover Pauli decomposition to Classiq PauliTerm list.

    Hamover format: [("X0*Z2", coeff), ("I", coeff), ...]
    Classiq format: sum of coeff * PAULI[qubit] products.
    """
    from classiq import Pauli
    from classiq.qmod.builtins.structs import IndexedPauli, SparsePauliOp, SparsePauliTerm

    pauli_map = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}

    sparse_terms: list[SparsePauliTerm] = []
    for pstring, coeff in pauli_terms:
        c = float(np.real(coeff))
        if abs(np.imag(coeff)) > imag_tol:
            import warnings

            warnings.warn(
                f"Dropping imaginary part {np.imag(coeff):.2e} of Pauli coeff "
                f"for term '{pstring}'",
                RuntimeWarning,
                stacklevel=2,
            )
        if abs(c) < 1e-15:
            continue

        # Build IndexedPauli list for non-identity axes
        indexed: list[IndexedPauli] = []
        if pstring == "I":
            # Identity term: no indexed paulis needed, just coefficient
            pass
        else:
            for token in pstring.split("*"):
                axis = token[0]
                qidx = int(token[1:])
                indexed.append(IndexedPauli(pauli=pauli_map[axis], index=qidx))

        sparse_terms.append(SparsePauliTerm(paulis=indexed, coefficient=c))

    return SparsePauliOp(terms=sparse_terms, num_qubits=n_qubits)


def compile_classiq_circuit(
    embedding: HamiltonianEmbedding,
    omega_func: Callable[[float], complex],
    Omega_func: Callable[[float], float],
    x: float,
    T: float,
    n_segments: int = 100,
    num_qdrift: int = 20,
):
    """Build a Classiq quantum program for the embedded search Hamiltonian.

    Returns the synthesized quantum program ready for execution.
    """
    from classiq import (
        Output,
        QArray,
        QBit,
        create_model,
        prepare_amplitudes,
        qdrift,
        qfunc,
        synthesize,
    )

    fields = SU2Fields(omega=omega_func, Omega=Omega_func, T=T)
    n_qubits = embedding.n_qubits

    # Compute initial state amplitudes
    psi0 = embedding.initial_state(x)
    amplitudes = np.real(psi0).tolist()

    # Check if H is time-independent (all segments produce same Pauli coefficients)
    t_mid = T / 2.0
    coeffs_0 = embedding.pauli_coefficients_at(fields, 0.0)
    coeffs_mid = embedding.pauli_coefficients_at(fields, t_mid)
    time_independent = len(coeffs_0) == len(coeffs_mid) and all(
        np.isclose(c1, c2, atol=1e-10)
        for (_, c1), (_, c2) in zip(coeffs_0, coeffs_mid)
    )

    if time_independent:
        # Single qDRIFT call for the full evolution
        pauli_op = _hamover_pauli_to_classiq(coeffs_0, n_qubits)

        @qfunc
        def main(q: Output[QArray[QBit]]) -> None:
            prepare_amplitudes(amplitudes, 0.0, q)
            qdrift(pauli_op, evolution_coefficient=T, num_qdrift=num_qdrift, qbv=q)
    else:
        # Piecewise-constant discretization: n_segments sequential qDRIFT calls
        dt = T / n_segments
        segment_ops = []
        for k in range(n_segments):
            t_k = (k + 0.5) * dt  # midpoint
            coeffs_k = embedding.pauli_coefficients_at(fields, t_k)
            op_k = _hamover_pauli_to_classiq(coeffs_k, n_qubits)
            segment_ops.append(op_k)

        @qfunc
        def main(q: Output[QArray[QBit]]) -> None:
            prepare_amplitudes(amplitudes, 0.0, q)
            for op_k in segment_ops:
                qdrift(op_k, evolution_coefficient=dt, num_qdrift=num_qdrift, qbv=q)

    model = create_model(main)
    quantum_program = synthesize(model)
    return quantum_program


def run_classiq_search(
    embedding: HamiltonianEmbedding,
    omega_func: Callable[[float], complex],
    Omega_func: Callable[[float], float],
    x: float,
    T: float,
    n_segments: int = 100,
    num_qdrift: int = 20,
    shots: int = 1000,
) -> ClassiqResult:
    """Compile, execute on Classiq, and decode measurement outcomes."""
    from classiq import execute, set_quantum_program_execution_preferences
    from classiq.execution import ExecutionPreferences

    quantum_program = compile_classiq_circuit(
        embedding=embedding,
        omega_func=omega_func,
        Omega_func=Omega_func,
        x=x,
        T=T,
        n_segments=n_segments,
        num_qdrift=num_qdrift,
    )

    # Execute
    preferences = ExecutionPreferences(num_shots=shots)
    quantum_program = set_quantum_program_execution_preferences(quantum_program, preferences)
    job = execute(quantum_program)
    results = job.result()
    parsed_counts = results[0].value.parsed_counts

    # Decode bitstrings
    found = 0
    not_found = 0
    leakage = 0
    all_bitstrings: list[str] = []

    for parsed_state in parsed_counts:
        qubit_vals = parsed_state.state["q"]
        count = parsed_state.shots
        # Convert list of qubit values [0, 1, ...] to bitstring "01..."
        bitstring = "".join(str(int(b)) for b in qubit_vals).zfill(embedding.n_qubits)
        for _ in range(count):
            all_bitstrings.append(bitstring)
            tag = embedding.decode(bitstring)
            if tag == "w":
                found += 1
            elif tag == "r":
                not_found += 1
            else:
                leakage += 1

    denom = found + not_found
    p_succ = float(found / denom) if denom > 0 else 0.0

    # Extract circuit metrics
    try:
        circuit_depth = results[0].value.physical_qubits_count  # fallback
        transpiled = quantum_program.transpiled_circuit
        circuit_depth = transpiled.depth if hasattr(transpiled, "depth") else 0
        cx_count = transpiled.count_ops.get("cx", 0) if hasattr(transpiled, "count_ops") else 0
    except Exception:
        circuit_depth = 0
        cx_count = 0

    return ClassiqResult(
        success_probability=p_succ,
        found_count=found,
        not_found_count=not_found,
        leakage_count=leakage,
        bitstrings=all_bitstrings,
        circuit_depth=circuit_depth,
        cx_count=cx_count,
        n_qubits=embedding.n_qubits,
        num_qdrift=num_qdrift,
    )
