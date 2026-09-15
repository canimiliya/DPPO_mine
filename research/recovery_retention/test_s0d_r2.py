from s0d_r2_core import gate_self_test, stable_seed


def test_gate_self_test():
    assert all(v["pass"] for v in gate_self_test().values())


def test_stable_seed_is_ordered_and_reproducible():
    assert stable_seed("square", 920001, "BC", "anchor") == stable_seed("square", 920001, "BC", "anchor")
    assert stable_seed("square", 920001, "BC", "anchor") != stable_seed("square", 920001, "FT", "anchor")
