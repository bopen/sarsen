import xarray as xr

from sarsen import apps


def test_simulate_acquisition(orbit_ds: xr.Dataset, dem_ecef: xr.DataArray) -> None:
    apps.simulate_acquisition(orbit_ds.position, dem_ecef)
