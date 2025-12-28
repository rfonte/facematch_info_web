import pytest


def test_skip_drive_integration():
    pytest.skip("Drive integration tests removed after revert")
