from pycgp import CGP
from .test_functions import build_func_lib


def test_normal(nin=1, nout=1):
    # print('Normal: %d, %d' % (nin, nout))
    library = build_func_lib()
    parent = CGP.random(nin, nout, 100, 1, library, 1.0, False)
    assert len(parent.genome) > 0
    parent.create_graph()
    for ni in range(len(parent.nodes)):
        n = parent.nodes[ni]
        for a in n.args:
            assert a >= 0
            assert a < (ni + nin)


def test_recursion(nin=1, nout=1):
    # print('Recursion: %d, %d' % (nin, nout))
    library = build_func_lib()
    parent = CGP.random(nin, nout, 100, 1, library, 2.0, False)
    assert len(parent.genome) > 0
    parent.create_graph()
    r_count = 0
    for ni in range(len(parent.nodes)):
        n = parent.nodes[ni]
        for a in n.args:
            assert a >= 0
            assert a <= len(parent.nodes) + nin
            if a >= ni + nin:
                r_count += 1
    print('Recursion connections: %d' % r_count)
    # assert r_count > 0


def test_all():
    for nin in range(1, 10):
        for nout in range(1, 10):
            test_normal(nin, nout)
            test_recursion(nin, nout)
