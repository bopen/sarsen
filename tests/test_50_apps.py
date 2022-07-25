import os
import pathlib

import numpy as np
import py
import pytest
import xarray as xr

from sarsen import apps

DATA_FOLDER = pathlib.Path(__file__).parent / "data"

DATA_PATHS = [
    DATA_FOLDER
    / "S1B_IW_GRDH_1SDV_20211223T051122_20211223T051147_030148_039993_5371.SAFE",
    DATA_FOLDER
    / "S1A_IW_SLC__1SDV_20220104T170557_20220104T170624_041314_04E951_F1F1.SAFE",
]

GROUPS = ["IW/VV", "IW1/VV"]

DEM_RASTER = DATA_FOLDER / "Rome-30m-DEM.tif"


def test_product_info() -> None:
    expected_geospatial_bbox = [
        11.86800305333565,
        40.87886713841886,
        15.32209672548896,
        42.78115380313222,
    ]

    res = apps.product_info(str(DATA_PATHS[0]))

    assert "product_type" in res
    assert np.allclose(res["geospatial_bbox"], expected_geospatial_bbox)


@pytest.mark.parametrize("data_path,group", zip(DATA_PATHS, GROUPS))
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_gtc(
    tmpdir: py.path.local,
    data_path: pathlib.Path,
    group: str,
) -> None:
    out = str(tmpdir.join("GTC.tif"))
    res = apps.terrain_correction(
        str(data_path),
        group,
        str(DEM_RASTER),
        output_urlpath=out,
    )

    assert isinstance(res, xr.DataArray)


@pytest.mark.parametrize("data_path,group", zip(DATA_PATHS, GROUPS))
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_fast_rtc(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str
) -> None:
    out = str(tmpdir.join("RTC.tif"))

    res = apps.terrain_correction(
        str(data_path),
        group,
        str(DEM_RASTER),
        correct_radiometry="gamma_nearest",
        output_urlpath=out,
    )

    assert isinstance(res, xr.DataArray)


@pytest.mark.parametrize("data_path,group", zip(DATA_PATHS, GROUPS))
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_rtc(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str
) -> None:
    out = str(tmpdir.join("RTC.tif"))

    res = apps.terrain_correction(
        str(data_path),
        group,
        str(DEM_RASTER),
        correct_radiometry="gamma_bilinear",
        output_urlpath=out,
    )

    assert isinstance(res, xr.DataArray)


@pytest.mark.parametrize("data_path,group", zip(DATA_PATHS, GROUPS))
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_gtc_dask(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str
) -> None:
    out = str(tmpdir.join("GTC.tif"))
    res = apps.terrain_correction(
        str(data_path),
        group,
        str(DEM_RASTER),
        output_urlpath=out,
        chunks=1024,
    )

    assert isinstance(res, xr.DataArray)


@pytest.mark.parametrize("data_path,group", zip(DATA_PATHS, GROUPS))
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_fast_rtc_dask(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str
) -> None:
    out = str(tmpdir.join("RTC.tif"))

    res = apps.terrain_correction(
        str(data_path),
        group,
        str(DEM_RASTER),
        correct_radiometry="gamma_nearest",
        output_urlpath=out,
        chunks=1024,
    )

    assert isinstance(res, xr.DataArray)


@pytest.mark.parametrize("data_path,group", zip(DATA_PATHS, GROUPS))
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_rtc_dask(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str
) -> None:
    out = str(tmpdir.join("RTC.tif"))

    res = apps.terrain_correction(
        str(data_path),
        group,
        str(DEM_RASTER),
        correct_radiometry="gamma_bilinear",
        output_urlpath=out,
        chunks=1024,
    )

    assert isinstance(res, xr.DataArray)
