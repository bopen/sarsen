from typing import Any

import pytest
from click.testing import CliRunner

from sarsen import __main__, apps, sentinel1

runner = CliRunner()

ARGS = ["product.SAFE", "IW/VV", "dem.tif"]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record the calls to the product and to the terrain correction."""
    calls: dict[str, Any] = {}

    def product(*args: Any) -> str:
        calls["product"] = args
        return "product"

    def terrain_correction(*args: Any, **kwargs: Any) -> None:
        calls["terrain_correction"] = (args, kwargs)

    monkeypatch.setattr(sentinel1, "Sentinel1SarProduct", product)
    monkeypatch.setattr(apps, "terrain_correction", terrain_correction)
    return calls


def test_main() -> None:
    res = runner.invoke(__main__.app, ["--version"])
    assert res.exit_code == 0
    assert res.stdout.startswith("sarsen, version ")

    res = runner.invoke(__main__.app, ["gtc", "--help"])
    assert res.exit_code == 0

    res = runner.invoke(__main__.app, ["rtc", "--help"])
    assert res.exit_code == 0


def test_gtc(calls: dict[str, Any]) -> None:
    res = runner.invoke(__main__.app, ["gtc", *ARGS])

    assert res.exit_code == 0, res.output
    assert calls["product"] == ("product.SAFE", "IW/VV")
    assert calls["terrain_correction"] == (
        ("product", "dem.tif"),
        {
            "output_urlpath": "GTC.tif",
            "enable_dask_distributed": False,
            "client_kwargs": {"processes": False},
            "chunks": 1024,
            "seed_step": None,
        },
    )

    res = runner.invoke(
        __main__.app,
        [
            "gtc",
            *ARGS,
            "--output-urlpath",
            "out.tif",
            "--enable-dask-distributed",
            "--client-kwargs-json",
            '{"processes": true}',
            "--chunks",
            "0",
            "--seed-step",
            "8",
        ],
    )

    assert res.exit_code == 0, res.output
    assert calls["terrain_correction"] == (
        ("product", "dem.tif"),
        {
            "output_urlpath": "out.tif",
            "enable_dask_distributed": True,
            "client_kwargs": {"processes": True},
            "chunks": None,
            "seed_step": (8, 8),
        },
    )


def test_stc(calls: dict[str, Any]) -> None:
    res = runner.invoke(__main__.app, ["stc", *ARGS])

    assert res.exit_code == 0, res.output
    assert calls["product"] == ("product.SAFE", "IW/VV")
    assert calls["terrain_correction"] == (
        ("product", "dem.tif"),
        {
            "output_urlpath": None,
            "correct_radiometry": "gamma_bilinear",
            "simulated_urlpath": "STC.tif",
            "grouping_area_factor": (3.0, 3.0),
            "enable_dask_distributed": False,
            "client_kwargs": {"processes": False},
            "chunks": 1024,
            "seed_step": None,
        },
    )

    res = runner.invoke(
        __main__.app,
        [
            "stc",
            *ARGS,
            "--simulated-urlpath",
            "sim.tif",
            "--grouping-area-factor",
            "2",
            "4.5",
            "--chunks",
            "512",
        ],
    )

    assert res.exit_code == 0, res.output
    assert calls["terrain_correction"] == (
        ("product", "dem.tif"),
        {
            "output_urlpath": None,
            "correct_radiometry": "gamma_bilinear",
            "simulated_urlpath": "sim.tif",
            "grouping_area_factor": (2.0, 4.5),
            "enable_dask_distributed": False,
            "client_kwargs": {"processes": False},
            "chunks": 512,
            "seed_step": None,
        },
    )


def test_rtc(calls: dict[str, Any]) -> None:
    res = runner.invoke(__main__.app, ["rtc", *ARGS])

    assert res.exit_code == 0, res.output
    assert calls["product"] == ("product.SAFE", "IW/VV")
    assert calls["terrain_correction"] == (
        ("product", "dem.tif"),
        {
            "correct_radiometry": "gamma_bilinear",
            "output_urlpath": "RTC.tif",
            "grouping_area_factor": (3.0, 3.0),
            "enable_dask_distributed": False,
            "client_kwargs": {"processes": False},
            "chunks": 1024,
            "seed_step": None,
        },
    )

    res = runner.invoke(
        __main__.app,
        [
            "rtc",
            *ARGS,
            "--output-urlpath",
            "out.tif",
            "--no-enable-dask-distributed",
            "--grouping-area-factor",
            "2",
            "4.5",
            "--seed-step",
            "8",
        ],
    )

    assert res.exit_code == 0, res.output
    assert calls["terrain_correction"] == (
        ("product", "dem.tif"),
        {
            "correct_radiometry": "gamma_bilinear",
            "output_urlpath": "out.tif",
            "grouping_area_factor": (2.0, 4.5),
            "enable_dask_distributed": False,
            "client_kwargs": {"processes": False},
            "chunks": 1024,
            "seed_step": (8, 8),
        },
    )
