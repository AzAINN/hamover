from hamover import HamoverSearch
from hamover.core import HBAR_NATURAL, SearchProblem
from hamover.embedding import encode_search
from hamover.hamiltonians.controlled import SU2Hamiltonian
from hamover.hamiltonians.protocol import SearchHamiltonian
from hamover.hamiltonians.scheduled import AdiabaticSearchHamiltonian
from hamover.hamiltonians.static import FarhiGutmannHamiltonian
from hamover.schedules.library import RolandCerfSchedule
from hamover.solver import closed_form_gqs, numerical_solve


def test_new_architecture_imports() -> None:
    assert HamoverSearch is not None
    assert SearchProblem is not None
    assert SearchHamiltonian is not None
    assert HBAR_NATURAL == 1.0
    assert callable(closed_form_gqs)
    assert callable(numerical_solve)
    assert callable(encode_search)
    assert SU2Hamiltonian is not None
    assert FarhiGutmannHamiltonian is not None
    assert AdiabaticSearchHamiltonian is not None
    assert RolandCerfSchedule is not None
