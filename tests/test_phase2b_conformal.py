from src.forecasting.conformal import conformal_bounds, finite_sample_quantile


def test_finite_sample_conformal_uses_capped_ceiling_rank():
    q,rank=finite_sample_quantile(list(range(1,9)),.80)
    assert rank==8 and q==8
    lo,hi,q,rank,valid=conformal_bounds(10,list(range(1,9)),.95,8)
    assert (lo,hi,q,rank,valid)==(2,18,8,8,True)
