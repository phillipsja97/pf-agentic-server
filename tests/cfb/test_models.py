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
