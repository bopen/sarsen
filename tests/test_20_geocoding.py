from typing import Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from sarsen import geocoding, orbit


def test_secant_method() -> None:
    def ufunc(
        t: npt.ArrayLike,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        ft: npt.NDArray[np.float64] = np.asarray(t).astype("float64")
        retval = 1.0 + 0.015 * ft - 0.0001 * ft**2 + 0.00003 * ft**3
        return retval, retval

    t_start: npt.NDArray[np.timedelta64] = np.array([np.timedelta64(-100, "ns")])

    # stop with df threshold
    res, _, _, _ = geocoding.secant_method(ufunc, -t_start, t_start, diff_ufunc=0.1)

    assert isinstance(res, np.ndarray)
    assert res.size == 1
    assert res[0] == np.timedelta64(-27, "ns")

    # stop with dt threshold
    res, _, _, _ = geocoding.secant_method(ufunc, -t_start, t_start, diff_ufunc=0.01)

    assert res[0] == np.timedelta64(-27, "ns")

    t_start = np.ones((2, 2), dtype="timedelta64[ns]") * 100  # type: ignore

    res, _, _, _ = geocoding.secant_method(ufunc, -t_start, t_start, diff_ufunc=0.1)

    assert res.shape == t_start.shape
    assert np.all(res == np.timedelta64(-27, "ns"))


def test_zero_doppler_plane_distance(
    dem_ecef: xr.DataArray, orbit_ds: xr.Dataset
) -> None:
    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(orbit_ds.position)

    res0, (res1, res2) = geocoding.zero_doppler_plane_distance(
        dem_ecef,
        orbit_interpolator.position(),
        orbit_interpolator.velocity(),
        orbit_ds.azimuth_time,
    )

    assert isinstance(res0, xr.DataArray)
    assert isinstance(res1, xr.DataArray)
    assert isinstance(res2, xr.DataArray)


def test_backward_geocode(dem_ecef: xr.DataArray, orbit_ds: xr.Dataset) -> None:
    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(orbit_ds.position)

    res = geocoding.backward_geocode(
        dem_ecef,
        orbit_interpolator.position(),
        orbit_interpolator.velocity(),
    )

    assert isinstance(res, xr.Dataset)
