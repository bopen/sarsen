"""Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019.

See: https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""

import functools
from typing import Any, Callable, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import xarray as xr

from . import orbit

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
        q: FloatArrayLike

        t_diff = t_curr - t_prev  # type: ignore

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

        q = f_curr - f_prev  # type: ignore

        # NOTE: in same cases f_curr * t_diff overflows datetime64[ns] before the division by q
        t_prev, t_curr = t_curr, t_curr - np.where(q != 0, f_curr / q, 0) * t_diff  # type: ignore
        f_prev = f_curr

    return t_curr, t_prev, f_curr, payload_curr


def newton_raphson_method(
    ufunc: Callable[[TimedeltaArrayLike], Tuple[FloatArrayLike, Any]],
    ufunc_prime: Callable[[TimedeltaArrayLike, Any], FloatArrayLike],
    t_curr: TimedeltaArrayLike,
    diff_ufunc: float = 1.0,
    diff_t: np.timedelta64 = np.timedelta64(0, "ns"),
) -> Tuple[TimedeltaArrayLike, FloatArrayLike, Any]:
    """Return the root of ufunc calculated using the Newton method."""
    # implementation based on https://en.wikipedia.org/wiki/Newton%27s_method
    # strong convergence, all points below one of the two thresholds
    while True:
        f_curr, payload_curr = ufunc(t_curr)

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        fp_curr = ufunc_prime(t_curr, payload_curr)

        t_diff = (f_curr / fp_curr) * np.timedelta64(10**9, "ns")  # type: ignore

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

        t_curr = t_curr - t_diff.rename("azimuth_time")

    return t_curr, f_curr, payload_curr


def zero_doppler_plane_distance_velocity(
    dem_ecef: xr.DataArray,
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    azimuth_time: xr.DataArray,
    dim: str = "axis",
) -> Tuple[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    dem_distance = dem_ecef - orbit_interpolator.position(azimuth_time)
    satellite_velocity = orbit_interpolator.velocity(azimuth_time)
    plane_distance_velocity = (dem_distance * satellite_velocity).sum(dim, skipna=False)
    return plane_distance_velocity, (dem_distance, satellite_velocity)


def zero_doppler_plane_distance_velocity_prime(
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    azimuth_time: xr.DataArray,
    payload: Tuple[xr.DataArray, xr.DataArray],
    dim: str = "axis",
) -> xr.DataArray:
    dem_distance, satellite_velocity = payload

    plane_distance_velocity_prime = (
        dem_distance * orbit_interpolator.acceleration(azimuth_time)
        - satellite_velocity**2
    ).sum(dim)
    return plane_distance_velocity_prime


def backward_geocode_simple(
    dem_ecef: xr.DataArray,
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    azimuth_time: xr.DataArray | None = None,
    dim: str = "axis",
    zero_doppler_distance: float = 1.0,
    satellite_speed: float = 7_500.0,
    method: str = "secant",
) -> tuple[xr.DataArray, xr.DataArray]:
    if azimuth_time is None:
        azimuth_time = orbit_interpolator.position().azimuth_time
    diff_ufunc = zero_doppler_distance * satellite_speed

    zero_doppler = functools.partial(
        zero_doppler_plane_distance_velocity, dem_ecef, orbit_interpolator
    )

    t_template = dem_ecef.isel({dim: 0}).drop_vars(dim).rename("azimuth_time")
    t_curr = xr.full_like(
        t_template,
        azimuth_time.values[azimuth_time.size // 2],
        dtype=azimuth_time.dtype,
    )

    if method == "secant":
        t_prev = xr.full_like(
            t_template, azimuth_time.values[0], dtype=azimuth_time.dtype
        )
        _, _, _, (dem_distance, satellite_velocity) = secant_method(
            zero_doppler,
            t_prev,
            t_curr,
            diff_ufunc,
        )
    elif method in {"newton", "newton_raphson"}:
        zero_doppler_prime = functools.partial(
            zero_doppler_plane_distance_velocity_prime, orbit_interpolator
        )
        _, _, (dem_distance, satellite_velocity) = newton_raphson_method(
            zero_doppler,
            zero_doppler_prime,
            t_curr,
            diff_ufunc,
        )
    # NOTE: dem_distance has the associated azimuth_time as a coordinate already
    return dem_distance, satellite_velocity


def backward_geocode(
    dem_ecef: xr.DataArray,
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    azimuth_time: xr.DataArray | None = None,
    dim: str = "axis",
    zero_doppler_distance: float = 1.0,
    satellite_speed: float = 7_500.0,
    method: str = "secant",
    seed_shape: tuple[int, int] | None = None,
) -> xr.Dataset:
    if seed_shape is not None:
        azimuth_time = azimuth_time

    dem_distance, satellite_velocity = backward_geocode_simple(
        dem_ecef,
        orbit_interpolator,
        azimuth_time,
        dim,
        zero_doppler_distance,
        satellite_speed,
        method,
    )

    acquisition = xr.Dataset(
        data_vars={
            "dem_distance": dem_distance,
            "satellite_velocity": satellite_velocity.transpose(*dem_distance.dims),
        }
    )
    return acquisition.reset_coords("azimuth_time")
