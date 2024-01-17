"""Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019.

See: https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""
import functools
from typing import Any, Callable, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import xarray as xr

TimedeltaArrayLike = TypeVar("TimedeltaArrayLike", bound=npt.ArrayLike)
FloatArrayLike = TypeVar("FloatArrayLike", bound=npt.ArrayLike)


def secant_method(
    ufunc: Callable[[TimedeltaArrayLike], Tuple[FloatArrayLike, FloatArrayLike]],
    t_prev: TimedeltaArrayLike,
    t_curr: TimedeltaArrayLike,
    diff_ufunc: float = 1.0,
    diff_t: np.timedelta64 = np.timedelta64(0, "ns"),
) -> Tuple[TimedeltaArrayLike, TimedeltaArrayLike, FloatArrayLike, Any]:
    """Return the root of ufunc calculated using the secant method."""
    # implementation modified from https://en.wikipedia.org/wiki/Secant_method
    f_prev, _ = ufunc(t_prev)

    # strong convergence, all points below one of the two thresholds
    while True:
        f_curr, payload_curr = ufunc(t_curr)

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

    return t_curr, t_prev, f_curr, payload_curr


# FIXME: interpolationg the direction decreses the precision, this function should
#   probably have velocity_ecef_sar in input instead
def zero_doppler_plane_distance(
    dem_ecef: xr.DataArray,
    position_ecef_sar: xr.DataArray,
    direction_ecef_sar: xr.DataArray,
    azimuth_time: TimedeltaArrayLike,
    dim: str = "axis",
) -> Tuple[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    dem_distance = dem_ecef - position_ecef_sar.interp(azimuth_time=azimuth_time)
    satellite_direction = direction_ecef_sar.interp(azimuth_time=azimuth_time)
    plane_distance = (dem_distance * satellite_direction).sum(dim, skipna=False)
    return plane_distance, (dem_distance, satellite_direction)


def backward_geocode(
    dem_ecef: xr.DataArray,
    position_ecef: xr.DataArray,
    velocity_ecef: xr.DataArray,
    azimuth_time: Optional[xr.DataArray] = None,
    dim: str = "axis",
    diff_ufunc: float = 1.0,
) -> xr.Dataset:
    direction_ecef = (
        velocity_ecef / xr.dot(velocity_ecef, velocity_ecef, dims=dim) ** 0.5
    )

    zero_doppler = functools.partial(
        zero_doppler_plane_distance, dem_ecef, position_ecef, direction_ecef
    )

    if azimuth_time is None:
        azimuth_time = position_ecef.azimuth_time
    t_template = dem_ecef.isel({dim: 0}).drop_vars(dim)
    t_prev = xr.full_like(t_template, azimuth_time.values[0], dtype=azimuth_time.dtype)
    t_curr = xr.full_like(t_template, azimuth_time.values[-1], dtype=azimuth_time.dtype)

    # NOTE: dem_distance has the associated azimuth_time as a coordinate already
    _, _, _, (dem_distance, satellite_direction) = secant_method(
        zero_doppler,
        t_prev,
        t_curr,
        diff_ufunc,
    )
    acquisition = xr.Dataset(
        data_vars={
            "dem_distance": dem_distance,
            "satellite_direction": satellite_direction.transpose(*dem_distance.dims),
        }
    )
    return acquisition.reset_coords("azimuth_time")
