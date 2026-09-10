import os
import pathlib

import py
import pytest
import xarray as xr

from sarsen import apps, sentinel1

DATA_FOLDER = pathlib.Path(__file__).parent / "data"

DATA_PATHS = [
    DATA_FOLDER
    / "S1B_IW_GRDH_1SDV_20211223T051122_20211223T051147_030148_039993_5371.SAFE",
    DATA_FOLDER
    / "S1A_IW_SLC__1SDV_20220104T170557_20220104T170624_041314_04E951_F1F1.SAFE",
    DATA_FOLDER
    / "S1A_IW_SLC__1SDV_20220104T170557_20220104T170624_041314_04E951_F1F1.SAFE",
    DATA_FOLDER
    / "S1A_S3_SLC__1SDV_20210401T152855_20210401T152914_037258_04638E_6001.SAFE",
]

GROUPS = ["IW/VV", "IW1/VV", "IW1/VV/8", "S3/VH"]

DEM_RASTERS = [
    DATA_FOLDER / "Rome-30m-DEM.tif",  # GRD IW/VV
    DATA_FOLDER / "Lazio-30m-DEM.tif",  # SLC IW1/VV
    DATA_FOLDER / "Lazio-30m-DEM.tif",  # SLC IW1/VV/8
    DATA_FOLDER / "Comoros-30m-DEM.tif",  # SLC S3/VH
]

DATA_PATH_GROUP_DEM = list(zip(DATA_PATHS, GROUPS, DEM_RASTERS))


@pytest.mark.parametrize("data_path,group,dem_raster", DATA_PATH_GROUP_DEM)
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_gtc(
    tmpdir: py.path.local,
    data_path: pathlib.Path,
    group: str,
    dem_raster: pathlib.Path,
) -> None:
    out = str(tmpdir.join("GTC.tif"))
    product = sentinel1.Sentinel1SarProduct(
        str(data_path),
        group,
        measurement_chunks=2048,
    )

    res = apps.terrain_correction(
        product,
        str(dem_raster),
        output_urlpath=out,
    )

    assert isinstance(res, xr.DataArray)
    assert "beta" in res.attrs["long_name"]
    assert res.notnull().any()


@pytest.mark.parametrize("data_path,group,dem_raster", DATA_PATH_GROUP_DEM)
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_fast_rtc(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str, dem_raster: pathlib.Path
) -> None:
    out = str(tmpdir.join("RTC.tif"))
    product = sentinel1.Sentinel1SarProduct(
        str(data_path),
        group,
        measurement_chunks=2048,
    )

    res = apps.terrain_correction(
        product,
        str(dem_raster),
        correct_radiometry="gamma_nearest",
        output_urlpath=out,
        seed_step=(32, 32),
    )

    assert isinstance(res, xr.DataArray)
    assert "gamma" in res.attrs["long_name"]
    assert res.notnull().any()


@pytest.mark.parametrize("data_path,group,dem_raster", DATA_PATH_GROUP_DEM)
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_rtc(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str, dem_raster: pathlib.Path
) -> None:
    out = str(tmpdir.join("RTC.tif"))
    product = sentinel1.Sentinel1SarProduct(
        str(data_path),
        group,
        measurement_chunks=2048,
    )

    res = apps.terrain_correction(
        product,
        str(dem_raster),
        correct_radiometry="gamma_bilinear",
        output_urlpath=out,
    )

    assert isinstance(res, xr.DataArray)
    assert "gamma" in res.attrs["long_name"]
    assert res.notnull().any()


@pytest.mark.parametrize("data_path,group,dem_raster", DATA_PATH_GROUP_DEM)
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_gtc_dask(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str, dem_raster: pathlib.Path
) -> None:
    out = str(tmpdir.join("GTC.tif"))
    product = sentinel1.Sentinel1SarProduct(
        str(data_path),
        group,
        measurement_chunks=2048,
    )

    res = apps.terrain_correction(
        product,
        str(dem_raster),
        output_urlpath=out,
        chunks=1024,
        seed_step=(32, 32),
    )

    assert isinstance(res, xr.DataArray)
    assert "beta" in res.attrs["long_name"]
    assert res.notnull().any()


@pytest.mark.parametrize("data_path,group,dem_raster", DATA_PATH_GROUP_DEM)
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_fast_rtc_dask(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str, dem_raster: pathlib.Path
) -> None:
    out = str(tmpdir.join("RTC.tif"))
    product = sentinel1.Sentinel1SarProduct(
        str(data_path),
        group,
        measurement_chunks=2048,
    )

    res = apps.terrain_correction(
        product,
        str(dem_raster),
        correct_radiometry="gamma_nearest",
        output_urlpath=out,
        chunks=1024,
        seed_step=(32, 32),
    )

    assert isinstance(res, xr.DataArray)
    assert "gamma" in res.attrs["long_name"]
    assert res.notnull().any()


@pytest.mark.parametrize("data_path,group,dem_raster", DATA_PATH_GROUP_DEM)
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="too much memory")
def test_terrain_correction_rtc_dask(
    tmpdir: py.path.local, data_path: pathlib.Path, group: str, dem_raster: pathlib.Path
) -> None:
    out = str(tmpdir.join("RTC.tif"))
    product = sentinel1.Sentinel1SarProduct(
        str(data_path),
        group,
        measurement_chunks=2048,
    )

    res = apps.terrain_correction(
        product,
        str(dem_raster),
        correct_radiometry="gamma_bilinear",
        output_urlpath=out,
        chunks=1024,
        seed_step=(32, 32),
    )

    assert isinstance(res, xr.DataArray)
    assert "gamma" in res.attrs["long_name"]
    assert res.notnull().any()
