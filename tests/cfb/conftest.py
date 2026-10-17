import pytest


@pytest.fixture(autouse=True)
def never_touch_the_predictions_table(monkeypatch):
    """No test may reach the real cfb_predictions table.

    run_slate persists what it serves, and a .env with MinIO credentials is
    enough for a test that only meant to check the slate's arithmetic to append
    fake rows to the live table. Blocked at the I/O boundary rather than at
    predictions.write, so the dedupe logic itself stays testable; tests that
    exercise the write patch over these.
    """
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the real cfb_predictions table")

    monkeypatch.setattr("workflows.cfb.predictions.write_deltalake", refuse)
    monkeypatch.setattr("workflows.cfb.predictions.DeltaTable", refuse)
