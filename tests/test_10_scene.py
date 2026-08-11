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


def test_upsample_coords(dem_raster: xr.DataArray) -> None:
    dem_raster_small = dem_raster.isel(x=slice(0, 3), y=slice(0, 2))
    expected_x = [
        12.44991,
        12.45,
        12.45009,
        12.45018,
        12.45028,
        12.45037,
        12.45046,
        12.45056,
        12.45065,
    ]
    expected_y = [
        41.95017,
        41.95022,
        41.95028,
        41.95033,
        41.95039,
        41.95044,
        41.9505,
        41.95056,
        41.95061,
        41.95067,
    ]
    res = scene.upsample_coords(dem_raster_small, x=3, y=5)

    assert res["x"].size == dem_raster_small.x.size * 3
    assert np.allclose(res["x"][1::3], dem_raster_small.x)
    assert np.allclose(res["x"], expected_x)

    assert res["y"].size == dem_raster_small.y.size * 5
    assert np.allclose(res["y"][2::5], dem_raster_small.y)
    assert np.allclose(res["y"], expected_y)


def test_upsample(dem_raster: xr.DataArray) -> None:
    res = scene.upsample(dem_raster, x=2)

    assert res.x.size == dem_raster.x.size * 2
    assert np.allclose(res.isel(x=0).mean(), 78.46527778)


def test_compute_dem_oriented_area(dem_raster: xr.DataArray) -> None:
    dem_3d = scene.convert_to_dem_3d(dem_raster)

    res = scene.compute_dem_oriented_area(dem_3d)

    assert set(res.dims) == {"axis", "y", "x"}
    assert res.name == "dem_oriented_area"
