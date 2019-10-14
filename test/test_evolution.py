from pycgp import CGP
from .test_functions import build_func_lib

def test_create_new():
    library = build_func_lib()
    parent = CGP.random(1, 1, 100, 1, library, 1.0, True)
    assert len(parent.genome) > 0

# TODO: evolution on memory problem (predict previous input)
