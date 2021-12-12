import os.path

import numpy as np
import pytest
import rioxarray
import xarray as xr

from sarsen import scene

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def dem_raster() -> xr.DataArray:
    dem_path = os.path.join(DATA_PATH, "Rome-30m-DEM.tif")
    dem_da: xr.DataArray
    dem_da = rioxarray.open_rasterio(dem_path).sel(band=1).drop_vars("band")
    return dem_da


def test_make_dem_3d(dem_raster: xr.DataArray) -> None:
    res = scene.make_dem_3d(dem_raster)

    assert res.dims == ("axis", "y", "x")
    assert res.name == "dem_3d"
    assert res.sel(x=12.5, y=42, method="nearest")[2] == 17.0
