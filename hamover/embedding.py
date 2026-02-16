"""Bridge embedding layer: map 2x2 su(2) search dynamics into multi-qubit subspaces.

This module is the core of the bridge pipeline:
2x2 control Hamiltonian -> encoded subspace dynamics -> Pauli decomposition -> backend compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable, Optional

import numpy as np

from hamover.core.types import SU2Fields


_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _normalize_state(vec: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vec / norm


def _to_dense_state(vec: np.ndarray, n_qubits: int, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=complex).reshape(-1)
    dim = 1 << n_qubits
    if arr.size != dim:
        raise ValueError(f"{name} must have length {dim}, got {arr.size}")
    return _normalize_state(arr, name)


def _omega_Omega_from_fields(fields: SU2Fields, t: float) -> tuple[complex, float]:
    return complex(fields.omega(t)), float(fields.Omega(t))


def all_pauli_labels(n_qubits: int) -> list[tuple[str, ...]]:
    """Return all n-qubit Pauli labels as tuples over {I,X,Y,Z}."""
    return list(product("IXYZ", repeat=n_qubits))


def label_to_pauli_string(label: Iterable[str]) -> str:
    """Encode Pauli label tuple into compact indexed format like X0*Z2."""
    terms = [f"{axis}{idx}" for idx, axis in enumerate(label) if axis != "I"]
    return "I" if not terms else "*".join(terms)


def pauli_string_to_label(pauli_string: str, n_qubits: int) -> tuple[str, ...]:
    """Decode compact indexed format into full Pauli label tuple."""
    if pauli_string == "I":
        return tuple("I" for _ in range(n_qubits))

    label = ["I"] * n_qubits
    for token in pauli_string.split("*"):
        axis = token[0]
        qidx = int(token[1:])
        if axis not in _PAULI_MATRICES:
            raise ValueError(f"Unknown Pauli axis '{axis}' in token '{token}'")
        if not (0 <= qidx < n_qubits):
            raise ValueError(f"Qubit index out of range in token '{token}'")
        label[qidx] = axis
    return tuple(label)


def pauli_label_to_matrix(label: Iterable[str]) -> np.ndarray:
    """Convert a full Pauli label tuple to its dense matrix."""
    mats = [_PAULI_MATRICES[a] for a in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def pauli_string_to_matrix(pauli_string: str, n_qubits: int) -> np.ndarray:
    """Convert compact Pauli string (X0*Z2) to dense matrix."""
    return pauli_label_to_matrix(pauli_string_to_label(pauli_string, n_qubits))


@dataclass(slots=True)
class HamiltonianEmbedding:
    """Encoded subspace embedding for an su(2) target Hamiltonian."""

    n_qubits: int
    w_encoded: np.ndarray
    r_encoded: np.ndarray
    penalty_strength: float = 0.0
    scaling_factor: float = 1.0
    target_index: Optional[int] = None
    scheme: str = "computational"
    _dim: int = field(init=False, repr=False)
    _eye: np.ndarray = field(init=False, repr=False)
    _P_w: np.ndarray = field(init=False, repr=False)
    _P_r: np.ndarray = field(init=False, repr=False)
    _P_wr: np.ndarray = field(init=False, repr=False)
    _P_rw: np.ndarray = field(init=False, repr=False)
    _P_sub: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >=1, got {self.n_qubits}")
        if self.penalty_strength < 0.0:
            raise ValueError(f"penalty_strength must be >=0, got {self.penalty_strength}")
        if self.scaling_factor <= 0.0:
            raise ValueError(f"scaling_factor must be >0, got {self.scaling_factor}")

        self.w_encoded = _to_dense_state(self.w_encoded, self.n_qubits, "w_encoded")
        self.r_encoded = _to_dense_state(self.r_encoded, self.n_qubits, "r_encoded")

        overlap = np.vdot(self.w_encoded, self.r_encoded)
        if not np.isclose(overlap, 0.0, atol=1e-10, rtol=0.0):
            raise ValueError(f"w_encoded and r_encoded must be orthogonal, got <w|r>={overlap}")

        self._dim = 1 << self.n_qubits
        self._eye = np.eye(self._dim, dtype=complex)
        self._P_w = np.outer(self.w_encoded, self.w_encoded.conj())
        self._P_r = np.outer(self.r_encoded, self.r_encoded.conj())
        self._P_wr = np.outer(self.w_encoded, self.r_encoded.conj())
        self._P_rw = np.outer(self.r_encoded, self.w_encoded.conj())
        self._P_sub = self._P_w + self._P_r

    @property
    def dimension(self) -> int:
        return self._dim

    def subspace_projector(self) -> np.ndarray:
        """Projector P_S = |w><w| + |r><r| for the encoded search subspace."""
        return self._P_sub.copy()

    def initial_state(self, x: float) -> np.ndarray:
        """Encoded initial state x|w> + sqrt(1-x^2)|r>."""
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0,1), got {x}")
        return x * self.w_encoded + np.sqrt(1.0 - x**2) * self.r_encoded

    def signal_hamiltonian(self, su2_fields: SU2Fields, t: float) -> np.ndarray:
        """Embedded signal Hamiltonian (without leakage penalty) at time t."""
        omega_t, Omega_t = _omega_Omega_from_fields(su2_fields, t)
        return self.scaling_factor * (
            omega_t * self._P_wr
            + np.conj(omega_t) * self._P_rw
            + Omega_t * (self._P_w - self._P_r)
        )

    def embedded_hamiltonian(self, su2_fields: SU2Fields, t: float) -> np.ndarray:
        """Full embedded Hamiltonian H_embed(t)=signal + penalty*(I-P_S)."""
        H_sig = self.signal_hamiltonian(su2_fields, t)
        H_pen = self.penalty_strength * (self._eye - self._P_sub)
        return H_sig + H_pen

    def embed(self, su2_fields: SU2Fields) -> Callable[[float], np.ndarray]:
        """Return callable H_embed(t)."""
        return lambda t: self.embedded_hamiltonian(su2_fields, float(t))

    def verify(self, su2_fields: SU2Fields, t_test: float, atol: float = 1e-8) -> float:
        """Return embedding projection error ||P H_embed P - H_signal||_F at t_test."""
        H_emb = self.embedded_hamiltonian(su2_fields, t_test)
        H_proj = self._P_sub @ H_emb @ self._P_sub
        H_target_full = self.signal_hamiltonian(su2_fields, t_test)
        err = float(np.linalg.norm(H_proj - H_target_full))
        if err > atol:
            return err
        return err

    def pauli_decompose(self, matrix: np.ndarray, tol: float = 1e-10) -> list[tuple[str, complex]]:
        """Dense n-qubit Pauli decomposition c_P = Tr(P M)/2^n."""
        mat = np.asarray(matrix, dtype=complex)
        if mat.shape != (self._dim, self._dim):
            raise ValueError(f"matrix must be {self._dim}x{self._dim}, got {mat.shape}")

        out: list[tuple[str, complex]] = []
        norm = float(1 << self.n_qubits)
        for label in all_pauli_labels(self.n_qubits):
            P = pauli_label_to_matrix(label)
            coeff = np.trace(P.conj().T @ mat) / norm
            if abs(coeff) > tol:
                out.append((label_to_pauli_string(label), complex(coeff)))
        return out

    def pauli_coefficients_at(
        self,
        su2_fields: SU2Fields,
        t: float,
        tol: float = 1e-10,
    ) -> list[tuple[str, complex]]:
        """Pauli decomposition of H_embed(t)."""
        return self.pauli_decompose(self.embedded_hamiltonian(su2_fields, t), tol=tol)

    def decode(self, bitstring: str, subspace_tol: float = 1e-6) -> str:
        """Decode computational-basis outcome as 'w', 'r', or 'leakage'.

        A bitstring belongs to the encoded subspace if it has nonzero overlap
        with |w⟩ or |r⟩ (i.e. p_w + p_r > subspace_tol).  Within the subspace,
        classify by whichever component dominates.  True leakage occurs only for
        basis states orthogonal to the entire {|w⟩, |r⟩} subspace (e.g. padding
        states when N is not a power of 2).
        """
        bits = bitstring.strip()
        if len(bits) != self.n_qubits or any(b not in "01" for b in bits):
            raise ValueError(f"bitstring must have length {self.n_qubits} over {{0,1}}, got '{bitstring}'")

        index = int(bits, 2)
        p_w = abs(self.w_encoded[index]) ** 2
        p_r = abs(self.r_encoded[index]) ** 2
        p_sub = p_w + p_r
        if p_sub < subspace_tol:
            return "leakage"
        if p_w >= p_r:
            return "w"
        return "r"

    def target_bitstring(self) -> Optional[str]:
        """Return encoded target bitstring if target index metadata is present."""
        if self.target_index is None:
            return None
        if not (0 <= self.target_index < self._dim):
            return None
        return format(self.target_index, f"0{self.n_qubits}b")


def encode_search(
    N: int,
    target_index: int,
    penalty_strength: float = 0.0,
    scaling_factor: float = 1.0,
) -> HamiltonianEmbedding:
    """Construct computational-basis embedding for search over N items."""
    if N < 2:
        raise ValueError(f"N must be >=2, got {N}")
    if not (0 <= target_index < N):
        raise ValueError(f"target_index must be in [0,{N-1}], got {target_index}")

    n_qubits = int(np.ceil(np.log2(N)))
    dim = 1 << n_qubits

    w = np.zeros(dim, dtype=complex)
    w[target_index] = 1.0

    s = np.zeros(dim, dtype=complex)
    s[:N] = 1.0 / np.sqrt(N)

    x = float(np.real(np.vdot(w, s)))
    r = (s - x * w) / np.sqrt(1.0 - x**2)

    return HamiltonianEmbedding(
        n_qubits=n_qubits,
        w_encoded=w,
        r_encoded=r,
        penalty_strength=penalty_strength,
        scaling_factor=scaling_factor,
        target_index=target_index,
        scheme="computational",
    )


def auto_penalty(
    su2_fields: SU2Fields,
    T: float,
    safety_factor: float = 5.0,
    n_samples: int = 256,
) -> float:
    """Choose penalty >= safety_factor * max_t max(|omega|,|Omega|)."""
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if safety_factor <= 0.0:
        raise ValueError(f"safety_factor must be positive, got {safety_factor}")

    ts = np.linspace(0.0, T, n_samples)
    max_field = 0.0
    for t in ts:
        omega_t, Omega_t = _omega_Omega_from_fields(su2_fields, float(t))
        max_field = max(max_field, abs(omega_t), abs(Omega_t))
    return float(safety_factor * max_field)


def dwave_embedding(N: int, target_index: int, penalty_strength: float = 0.0) -> HamiltonianEmbedding:
    """Factory for D-Wave-oriented embedding metadata (computational baseline)."""
    emb = encode_search(N=N, target_index=target_index, penalty_strength=penalty_strength)
    emb.scheme = "dwave"
    return emb


def ionq_embedding(N: int, target_index: int, penalty_strength: float = 0.0) -> HamiltonianEmbedding:
    """Factory for IonQ-oriented embedding metadata (computational baseline)."""
    emb = encode_search(N=N, target_index=target_index, penalty_strength=penalty_strength)
    emb.scheme = "ionq"
    return emb


def quera_embedding(N: int, target_index: int, penalty_strength: float = 0.0) -> HamiltonianEmbedding:
    """Factory for QuEra-oriented embedding metadata (computational baseline)."""
    emb = encode_search(N=N, target_index=target_index, penalty_strength=penalty_strength)
    emb.scheme = "quera"
    return emb
