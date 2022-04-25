import xarray as xr

from sarsen import radiometry


def test_compute_gamma_area(dem_ecef: xr.DataArray) -> None:
    dem_direction = xr.DataArray()
    res = radiometry.compute_gamma_area(dem_ecef, dem_direction)

    assert isinstance(res, xr.DataArray)
