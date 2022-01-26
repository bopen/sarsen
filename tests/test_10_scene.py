import os.path

import numpy as np
import pytest
import xarray as xr

from sarsen import scene

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def dem_raster() -> xr.DataArray:
    dem_path = os.path.join(DATA_PATH, "Rome-30m-DEM.tif")
    dem_da: xr.DataArray
    dem_da = xr.open_dataarray(dem_path, engine="rasterio")  # type: ignore
    dem_da = dem_da.squeeze(drop=True)
    return dem_da


def test_convert_to_dem_3d(dem_raster: xr.DataArray) -> None:
    res = scene.convert_to_dem_3d(dem_raster)

    assert res.dims == ("axis", "y", "x")
    assert res.name == "dem_3d"
    assert res.sel(x=12.5, y=42, method="nearest")[2] == 17.0


def test_transform_dem_3d(dem_raster: xr.DataArray) -> None:
    dem_3d = scene.convert_to_dem_3d(dem_raster)

    # from height over the geoid to height over the ellipsoid
    res = scene.transform_dem_3d(dem_3d, "EPSG:4979", source_crs=dem_3d.rio.crs)

    assert res.dims == ("axis", "y", "x")
    # this assert fails if proj-data is not properly installed on the system
    assert abs(res.sel(x=12.5, y=42, method="nearest")[2] - 65.613) < 0.001

    expected = [4634523.742, 1027449.178, 4245647.74]

    # from geographic to geocentric (ECEF)
    res = scene.transform_dem_3d(dem_3d, "EPSG:4978")

    assert res.dims == ("axis", "y", "x")
    assert np.allclose(
        res.sel(x=12.5, y=42, method="nearest"), expected, rtol=0, atol=0.001
    )
