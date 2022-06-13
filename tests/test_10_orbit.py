import numpy as np
import xarray as xr

from sarsen import orbit


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

    res = orbit_interpolator.position()
    assert res.dims == ("azimuth_time", "axis")

    res = orbit_interpolator.velocity()
    assert res.dims == ("azimuth_time", "axis")


def test_OrbitPolyfitIterpolator_timedelta64(orbit_ds: xr.Dataset) -> None:
    position = orbit_ds.data_vars["position"]
    position = position.assign_coords(
        azimuth_time=position.azimuth_time - position.azimuth_time[0]
    )
    epoch = position.azimuth_time.values[0]
    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(
        position, epoch=epoch, deg=4
    )

    res = orbit_interpolator.position(position.azimuth_time)

    assert res.dims == ("azimuth_time", "axis")
    assert np.allclose(res, position, rtol=0, atol=0.001)

    expected_velocity = orbit_ds.data_vars["velocity"]
    res = orbit_interpolator.velocity(position.azimuth_time)

    assert res.dims == ("azimuth_time", "axis")
    assert np.allclose(res, expected_velocity, rtol=0, atol=0.02)
