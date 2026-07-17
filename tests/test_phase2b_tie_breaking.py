from src.tuning.forward_search import complexity_key


def test_tie_breaking_prefers_lower_svr_capacity():
    low={"C":.1,"epsilon":.01,"gamma":"scale"}; high={"C":10,"epsilon":.01,"gamma":"scale"}
    assert complexity_key("SVR",low)<complexity_key("SVR",high)
