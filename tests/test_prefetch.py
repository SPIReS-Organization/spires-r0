import time, pytest
from spires_r0.prefetch import one_ahead

def test_order_and_single_load():
    calls = []
    out = list(one_ahead([1, 2, 3], lambda i: (calls.append(i), i * 10)[1]))
    assert out == [(1, 10), (2, 20), (3, 30)] and sorted(calls) == [1, 2, 3]

def test_empty():
    assert list(one_ahead([], lambda x: x)) == []

def test_overlaps_load_and_consume():
    def load(i): time.sleep(0.05); return i
    t0 = time.perf_counter()
    for _ in one_ahead([1, 2, 3, 4], load): time.sleep(0.05)
    assert time.perf_counter() - t0 < 0.05 * 8 * 0.85

def test_propagates_loader_error():
    def load(i):
        if i == 2: raise RuntimeError("boom")
        return i
    g = one_ahead([1, 2], load); assert next(g) == (1, 1)
    with pytest.raises(RuntimeError, match="boom"): next(g)
