from temporal_nlg.evaluation.m3_e2_human_loop import coerce_unit_score


def test_coerce_unit_score_accepts_unit_interval():
    assert coerce_unit_score(0.0) == 0.0
    assert coerce_unit_score(1.0) == 1.0
    assert coerce_unit_score(0.25) == 0.25


def test_coerce_unit_score_maps_likert_1_to_5():
    assert coerce_unit_score(1) == 0.0
    assert coerce_unit_score(3) == 0.5
    assert coerce_unit_score(5) == 1.0


def test_coerce_unit_score_handles_strings_and_bools():
    assert coerce_unit_score("0.8") == 0.8
    assert coerce_unit_score(True) == 1.0
    assert coerce_unit_score(False) == 0.0
    assert coerce_unit_score("not a number") is None
