import os
import pathlib
from typing import Callable

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


def test_compute_chunk_1d() -> None:

    ext_chunks, ext_chunks_bound, int_chunks = apps.compute_chunks_1d(
        dim_size=20, chunks=10, bound=2
    )
    assert ext_chunks == [slice(0, 12), slice(8, 20)]
    assert int_chunks == [slice(0, 10), slice(10, 20)]
    assert ext_chunks_bound == [slice(0, 10), slice(2, 12)]

    ext_chunks, ext_chunks_bound, int_chunks = apps.compute_chunks_1d(
        dim_size=13, chunks=10, bound=2
    )
    assert ext_chunks == [slice(0, 12), slice(8, 13)]
    assert int_chunks == [slice(0, 10), slice(10, 13)]
    assert ext_chunks_bound == [slice(0, 10), slice(2, 5)]

    # check bound case: k * chunks + bound == dim_size
    ext_chunks, ext_chunks_bound, int_chunks = apps.compute_chunks_1d(
        dim_size=22, chunks=10, bound=2
    )
    assert ext_chunks == [slice(0, 12), slice(8, 22)]
    assert int_chunks == [slice(0, 10), slice(10, 22)]
    assert ext_chunks_bound == [slice(0, 10), slice(2, 14)]

    # check bound case: k * chunks + bound > dim_size >  k * chunks
    ext_chunks, ext_chunks_bound, int_chunks = apps.compute_chunks_1d(
        dim_size=21, chunks=10, bound=2
    )
    assert ext_chunks == [slice(0, 12), slice(8, 21)]
    assert int_chunks == [slice(0, 10), slice(10, 21)]
    assert ext_chunks_bound == [slice(0, 10), slice(2, 13)]

    # check bound case: dim_size < bound
    ext_chunks, ext_chunks_bound, int_chunks = apps.compute_chunks_1d(
        dim_size=2, chunks=10, bound=10
    )
    assert ext_chunks == [slice(0, 2)]
    assert int_chunks == [slice(0, 2)]
    assert ext_chunks_bound == [slice(0, 2)]


def test_compute_chunks() -> None:

    ext_chunks, ext_chunks_bound, int_chunks = apps.compute_chunks(
        dims={"x": 10, "y": 21}, chunks=10, bound=2
    )
    assert {"x": slice(0, 10), "y": slice(0, 12)} in ext_chunks
    assert {"x": slice(0, 10), "y": slice(8, 21)} in ext_chunks

    assert {"x": slice(0, 10), "y": slice(0, 10)} in int_chunks
    assert {"x": slice(0, 10), "y": slice(10, 21)} in int_chunks

    assert {"x": slice(0, 10), "y": slice(0, 10)} in ext_chunks_bound
    assert {"x": slice(0, 10), "y": slice(2, 13)} in ext_chunks_bound


def test_execute_on_overlapping_blocks() -> None:
    arr = xr.DataArray(np.arange(22 * 31).reshape((22, 31)), dims=("x", "y"))
    function: Callable[[xr.DataArray], xr.DataArray] = lambda x: x
    res = apps.execute_on_overlapping_blocks(
        function=function, obj=arr, chunks=10, bound=2
    )
    res == arr
