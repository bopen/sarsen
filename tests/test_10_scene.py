import numpy as np
import pytest
import xarray as xr

from sarsen import scene


def test_convert_to_dem_3d(dem_raster: xr.DataArray) -> None:
    res = scene.convert_to_dem_3d(dem_raster)

    assert res.dims == ("axis", "y", "x")
    assert res.name == "dem_3d"
    assert res.sel(x=12.5, y=42, method="nearest")[2] == 17.0


@pytest.mark.xfail()
def test_transform_dem_3d(dem_raster: xr.DataArray) -> None:
    dem_3d = scene.convert_to_dem_3d(dem_raster)

    # from height over the geoid to height over the ellipsoid
    res = scene.transform_dem_3d(dem_3d, dem_3d.rio.crs, "EPSG:4979")

    assert res.dims == ("axis", "y", "x")
    # this assert fails if proj-data is not properly installed on the system
    assert abs(res.sel(x=12.5, y=42, method="nearest")[2] - 65.613) < 0.001

    expected = [4634523.742, 1027449.178, 4245647.74]

    # from geographic to geocentric (ECEF)
    res = scene.transform_dem_3d(dem_3d, dem_3d.rio.crs, "EPSG:4978")

    assert res.dims == ("axis", "y", "x")
    assert np.allclose(
        res.sel(x=12.5, y=42, method="nearest"), expected, rtol=0, atol=0.001
    )


def test_upsample(dem_raster: xr.DataArray) -> None:
    res = scene.upsample(dem_raster, x=2)

    assert res.x.size == dem_raster.x.size * 2
    assert np.allclose(res.isel(x=0).mean(), 78.46527778)


def test_compute_dem_oriented_area(dem_raster: xr.DataArray) -> None:
    dem_3d = scene.convert_to_dem_3d(dem_raster)

    res = scene.compute_dem_oriented_area(dem_3d)

    assert set(res.dims) == {"axis", "y", "x"}
    assert res.name == "dem_oriented_area"
