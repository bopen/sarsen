import pytest
from typer.testing import CliRunner

from sarsen import __main__

runner = CliRunner()


@pytest.mark.xfail()
def test_main() -> None:
    res = runner.invoke(__main__.app, ["gtc", "--help"])
    assert res.exit_code == 0

    res = runner.invoke(__main__.app, ["rtc", "--help"])
    assert res.exit_code == 0
