import os

import pytest
import xarray as xr

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def orbit_ds() -> xr.Dataset:
    # orbit data are from a Sentinel-1 product
    ds = xr.Dataset(
        {
            "position": (
                ("azimuth_time", "axis"),
                [
                    [4299854.769, 1453596.443, 5418885.179],
                    [4359238.173, 1452560.406, 5371628.586],
                    [4418131.478, 1451275.368, 5323765.698],
                    [4476527.709, 1449742.188, 5275301.901],
                    [4534419.947, 1447961.762, 5226242.648],
                    [4591801.329, 1445935.027, 5176593.459],
                ],
            ),
            "velocity": (
                ("azimuth_time", "axis"),
                [
                    [5962.611698, -91.122756, -4695.177565],
                    [5913.952956, -116.0645, -4756.073476],
                    [5864.593853, -140.922246, -4816.434357],
                    [5814.540013, -165.692188, -4876.253356],
                    [5763.79714, -190.370539, -4935.523681],
                    [5712.371027, -214.95353, -4994.238602],
                ],
            ),
        },
        coords={
            "azimuth_time": (
                "azimuth_time",
                [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0],
                {"units": "seconds since 2021-04-01 05:25:29"},
            ),
            "axis": ("axis", [0, 1, 2]),
        },
    )
    return xr.decode_cf(ds)


@pytest.fixture
def dem_raster() -> xr.DataArray:
    from sarsen import scene

    dem_path = os.path.join(DATA_PATH, "Rome-30m-DEM.tif")
    return scene.open_dem_raster(dem_path)


@pytest.fixture
def dem_ecef(dem_raster: xr.DataArray) -> xr.DataArray:
    from sarsen import scene

    return scene.convert_to_dem_ecef(dem_raster, source_crs=dem_raster.rio.crs)
