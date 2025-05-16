import numpy as np
import xarray as xr

from sarsen import orbit


def test_orbit_time_to_azimuth_time() -> None:
    epoch = np.datetime64("2025-05-16T08:38:12.123456789", "ns")

    res = orbit.orbit_time_to_azimuth_time(xr.DataArray(0.0), epoch)

    assert res == epoch

    res = orbit.orbit_time_to_azimuth_time(xr.DataArray(-120.2), epoch)

    assert res == epoch - np.timedelta64(120200, "ms")

    res = orbit.orbit_time_to_azimuth_time(xr.DataArray(1e-10), epoch)

    assert res == epoch


def test_azimuth_time_to_orbit_time() -> None:
    epoch = np.datetime64("2025-05-16T08:38:12.123456789", "ns")
    seconds = -120.43256234
    date = orbit.orbit_time_to_azimuth_time(xr.DataArray(seconds), epoch)

    res = orbit.azimuth_time_to_orbit_time(date, epoch)

    assert res == seconds


def test_OrbitPolyfitInterpolator_datetime64(orbit_ds: xr.Dataset) -> None:
    position = orbit_ds.data_vars["position"]
    orbit_interpolator = orbit.OrbitPolyfitInterpolator.from_position(position, deg=4)

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


def test_OrbitPolyfitInterpolator_timedelta64(orbit_ds: xr.Dataset) -> None:
    position = orbit_ds.data_vars["position"]
    position = position.assign_coords(
        azimuth_time=position.azimuth_time - position.azimuth_time[0]
    )
    epoch = position.azimuth_time.values[0]
    orbit_interpolator = orbit.OrbitPolyfitInterpolator.from_position(
        position, epoch=epoch, deg=4
    )

    res = orbit_interpolator.position(position.azimuth_time)

    assert res.dims == ("azimuth_time", "axis")
    assert np.allclose(res, position, rtol=0, atol=0.001)

    expected_velocity = orbit_ds.data_vars["velocity"]
    res = orbit_interpolator.velocity(position.azimuth_time)

    assert res.dims == ("azimuth_time", "axis")
    assert np.allclose(res, expected_velocity, rtol=0, atol=0.02)


def test_OrbitPolyfitInterpolator_values(orbit_ds: xr.Dataset) -> None:
    position = orbit_ds.data_vars["position"]
    orbit_interpolator = orbit.OrbitPolyfitInterpolator.from_position(position, deg=4)

    distance = np.linalg.norm(orbit_interpolator.position(), axis=1)

    assert np.allclose(distance, 7_068_663, rtol=0.001)

    velocity = np.linalg.norm(orbit_interpolator.velocity(), axis=1)

    assert np.allclose(velocity, 7_590, rtol=0.001)

    acceleration = np.linalg.norm(orbit_interpolator.acceleration(), axis=1)

    assert np.allclose(acceleration, 8.18, rtol=0.001)
