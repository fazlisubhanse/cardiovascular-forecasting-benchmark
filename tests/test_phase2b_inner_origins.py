from src.tuning.inner_origins import valid_inner_origins


def test_inner_targets_are_inside_outer_history():
    origins=valid_inner_origins(1970,2005,5,8,20)
    assert len(origins)==8 and max(o+5 for o in origins)<=2005 and origins==sorted(origins)
