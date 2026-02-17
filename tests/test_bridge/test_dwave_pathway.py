from hamover.backends.dwave_backend import run_simulated_annealing
from hamover.embedding import encode_search


def test_dwave_simulated_annealing_pathway() -> None:
    embedding = encode_search(N=2, target_index=0, penalty_strength=0.0)

    omega = lambda _t: 0.0 + 0.0j
    Omega = lambda _t: -1.0

    result = run_simulated_annealing(
        embedding=embedding,
        omega_func=omega,
        Omega_func=Omega,
        T=1.0,
        shots=200,
        rng_seed=7,
    )

    assert result.ising_model.dropped_terms == 0
    assert 0.0 <= result.success_probability <= 1.0
    assert result.success_probability > 0.5
    assert len(result.bitstrings) == 200
