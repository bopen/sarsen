from typing import Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from sarsen import geocoding, orbit


def test_secant_method() -> None:
    def ufunc(
        t: npt.ArrayLike,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        retval = 1.0 + 0.015 * t - 0.0001 * t**2 + 0.00003 * t**3  # type: ignore
        return retval, retval  # type: ignore

    t_start = np.array([-100.0])

    # stop with df threshold
    res, _, _, _, _ = geocoding.secant_method(ufunc, -t_start, t_start, diff_ufunc=0.1)

    assert isinstance(res, np.ndarray)
    assert res.size == 1
    assert np.allclose(res, -25.1724)

    # stop with dt threshold
    res, _, _, _, _ = geocoding.secant_method(ufunc, -t_start, t_start, diff_ufunc=0.01)

    assert np.allclose(res, -26.3065)

    t_start = np.ones((2, 2))

    res, _, _, _, _ = geocoding.secant_method(ufunc, -t_start, t_start, diff_ufunc=0.01)

    assert res.shape == t_start.shape
    assert np.allclose(res, -26.102)


def test_zero_doppler_plane_distance_velocity(
    dem_ecef: xr.DataArray, orbit_ds: xr.Dataset
) -> None:
    orbit_interpolator = orbit.OrbitPolyfitInterpolator.from_position(orbit_ds.position)

    res0, (res1, res2) = geocoding.zero_doppler_plane_distance_velocity(
        dem_ecef, orbit_interpolator, orbit_ds.azimuth_time
    )

    assert isinstance(res0, xr.DataArray)
    assert isinstance(res1, xr.DataArray)
    assert isinstance(res2, xr.DataArray)


def test_backward_geocode(dem_ecef: xr.DataArray, orbit_ds: xr.Dataset) -> None:
    orbit_interpolator = orbit.OrbitPolyfitInterpolator.from_position(orbit_ds.position)

    res = geocoding.backward_geocode(dem_ecef, orbit_interpolator)

    assert isinstance(res, xr.Dataset)

    res = geocoding.backward_geocode(dem_ecef, orbit_interpolator, method="newton")

    assert isinstance(res, xr.Dataset)
