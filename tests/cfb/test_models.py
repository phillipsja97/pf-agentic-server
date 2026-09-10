from schemas.models import CfbAnalystRequest


def test_cfb_analyst_request_defaults_power4_true():
    req = CfbAnalystRequest(question="who led the SEC in rushing?")
    assert req.question == "who led the SEC in rushing?"
    assert req.power4_only is True


def test_cfb_analyst_request_power4_false():
    req = CfbAnalystRequest(question="all FBS rushing leaders", power4_only=False)
    assert req.power4_only is False


def test_cfb_analyst_request_requires_question():
    import pytest
    with pytest.raises(Exception):
        CfbAnalystRequest()


from schemas.models import CfbMatchupRequest, CfbEdge


def test_cfb_matchup_request_defaults():
    req = CfbMatchupRequest(home_team="Georgia", away_team="Alabama")
    assert req.neutral_site is False
    assert req.season is None and req.through_week is None


def test_cfb_edge_favors_is_constrained():
    CfbEdge(stat="points", home_value=30.0, away_value=20.0, diff=10.0, favors="home", unit="pts")
    import pytest
    with pytest.raises(Exception):
        CfbEdge(stat="points", home_value=30.0, away_value=20.0, diff=10.0, favors="left", unit="pts")
