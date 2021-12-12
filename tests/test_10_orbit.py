import numpy as np
import pytest
import xarray as xr

from sarsen import orbit


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
    return xr.decode_cf(ds)  # type: ignore


def test_OrbitPolyfitIterpolator_datetime64(orbit_ds: xr.Dataset) -> None:
    position = orbit_ds.data_vars["position"]
    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(position, deg=4)

    res = orbit_interpolator.position(position.azimuth_time)

    assert res.dims == ("azimuth_time", "axis")
    assert np.allclose(res, position, rtol=0, atol=0.001)

    expected_velocity = orbit_ds.data_vars["velocity"]
    res = orbit_interpolator.velocity(position.azimuth_time)

    assert res.dims == ("azimuth_time", "axis")
    assert np.allclose(res, expected_velocity, rtol=0, atol=0.02)


def test_OrbitPolyfitIterpolator_timedelta64(orbit_ds: xr.Dataset) -> None:
    position = orbit_ds.data_vars["position"]
    position = position.assign_coords(azimuth_time=position.azimuth_time - position.azimuth_time[0])  # type: ignore
    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(position, deg=4)

    res = orbit_interpolator.position(position.azimuth_time)

    assert res.dims == ("azimuth_time", "axis")
    assert np.allclose(res, position, rtol=0, atol=0.001)

    expected_velocity = orbit_ds.data_vars["velocity"]
    res = orbit_interpolator.velocity(position.azimuth_time)

    assert res.dims == ("azimuth_time", "axis")
    assert np.allclose(res, expected_velocity, rtol=0, atol=0.02)
