from vlm_eval.metrics import bow_scores


def test_bow_scores_basic_overlap():
    scores = bow_scores("hej världen världen", "hej världen")
    assert scores.precision == 1.0
    assert scores.recall == 2 / 3


def test_bow_scores_empty():
    scores = bow_scores("", "")
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0
