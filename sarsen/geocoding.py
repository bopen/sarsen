"""
Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019
https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""
import functools
import typing as T

import numpy as np
import numpy.typing as npt
import xarray as xr

SPEED_OF_LIGHT = 299_792_458  # m / s

TimedeltaArrayLike = T.TypeVar("TimedeltaArrayLike", bound=npt.ArrayLike)
FloatArrayLike = T.TypeVar("FloatArrayLike", bound=npt.ArrayLike)


def secant_method(
    ufunc: T.Callable[[TimedeltaArrayLike], T.Tuple[FloatArrayLike, FloatArrayLike]],
    t_prev: TimedeltaArrayLike,
    t_curr: TimedeltaArrayLike,
    diff_ufunc: float = 1,
    diff_t: np.timedelta64 = np.timedelta64(0, "ns"),
) -> T.Tuple[TimedeltaArrayLike, TimedeltaArrayLike, FloatArrayLike, FloatArrayLike]:
    """Return the root of ufunc calculated using the secant method."""
    # implementation modified from https://en.wikipedia.org/wiki/Secant_method
    f_prev, _ = ufunc(t_prev)

    # strong convergence, all points below one of the two thresholds
    while True:
        f_curr, g_curr = ufunc(t_curr)

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        t_diff: TimedeltaArrayLike
        p: TimedeltaArrayLike
        q: FloatArrayLike

        t_diff = t_curr - t_prev  # type: ignore
        p = f_curr * t_diff  # type: ignore
        q = f_curr - f_prev  # type: ignore

        # t_prev, t_curr = t_curr, t_curr - f_curr * np.timedelta64(-148_000, "ns")
        t_prev, t_curr = t_curr, t_curr - np.where(q != 0, p / q, 0)  # type: ignore
        f_prev = f_curr

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

    return t_curr, t_prev, f_curr, g_curr


def zero_doppler_plane_distance(
    dem_ecef: xr.DataArray,
    position_ecef_sar: xr.DataArray,
    direction_ecef_sar: xr.DataArray,
    azimuth_time: TimedeltaArrayLike,
    dim: str = "axis",
) -> T.Tuple[xr.DataArray, xr.DataArray]:
    distance = dem_ecef - position_ecef_sar.interp(azimuth_time=azimuth_time)
    plane_distance = (
        distance * direction_ecef_sar.interp(azimuth_time=azimuth_time)
    ).sum(dim, skipna=False)
    return plane_distance, distance


def backward_geocode(
    dem_ecef: xr.DataArray,
    data_sar_time: xr.DataArray,
    position_ecef_sar: xr.DataArray,
    velocity_ecef_sar: xr.DataArray,
    dim: str = "axis",
    diff_ufunc: float = 1.0,
) -> xr.Dataset:
    direction_ecef_sar = velocity_ecef_sar / np.sqrt((velocity_ecef_sar ** 2).sum(dim))

    zero_doppler = functools.partial(
        zero_doppler_plane_distance, dem_ecef, position_ecef_sar, direction_ecef_sar
    )

    t_template = dem_ecef.isel(axis=0).drop_vars("axis").astype(data_sar_time.dtype)
    t_prev = xr.full_like(t_template, data_sar_time.values[0])
    t_curr = xr.full_like(t_template, data_sar_time.values[-1])

    dem_time, _, _, dem_distance = secant_method(
        zero_doppler, t_prev, t_curr, diff_ufunc
    )
    dem_slant_range = (dem_distance ** 2).sum(dim) ** 0.5
    dem_slant_range = dem_slant_range.drop_vars(["azimuth_time", "line"])
    if "azimuth_time" in dem_time.coords:
        dem_time = dem_time.drop_vars("azimuth_time")
    if "line" in dem_time.coords:
        dem_time = dem_time.drop_vars("line")

    return xr.merge(
        [
            dem_time.rename("azimuth_time"),
            dem_slant_range.rename("slant_range_time") * 2 / SPEED_OF_LIGHT,
        ]
    )