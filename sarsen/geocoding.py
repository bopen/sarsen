"""Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019.

See: https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""

import functools
from typing import Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt
import xarray as xr

from . import orbit

ArrayLike = TypeVar("ArrayLike", bound=npt.ArrayLike)
FloatArrayLike = TypeVar("FloatArrayLike", bound=npt.ArrayLike)


def secant_method(
    ufunc: Callable[[ArrayLike], tuple[FloatArrayLike, Any]],
    t_prev: ArrayLike,
    t_curr: ArrayLike,
    diff_ufunc: float = 1.0,
    diff_t: Any = 1e-6,
    maxiter: int = 10,
) -> tuple[ArrayLike, ArrayLike, FloatArrayLike, int, Any]:
    """Return the root of ufunc calculated using the secant method."""
    # implementation modified from https://en.wikipedia.org/wiki/Secant_method
    f_prev, _ = ufunc(t_prev)

    # strong convergence, all points below one of the two thresholds
    for k in range(maxiter):
        f_curr, payload_curr = ufunc(t_curr)

        # print(f"{f_curr / 7500}")

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        t_diff = t_curr - t_prev  # type: ignore

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

        q = f_curr - f_prev  # type: ignore

        # NOTE: in same cases f_curr * t_diff overflows datetime64[ns] before the division by q
        t_prev, t_curr = t_curr, t_curr - np.where(q != 0, f_curr / q, 0) * t_diff  # type: ignore
        f_prev = f_curr

    return t_curr, t_prev, f_curr, k, payload_curr


def newton_raphson_method(
    ufunc: Callable[[ArrayLike], tuple[FloatArrayLike, Any]],
    ufunc_prime: Callable[[ArrayLike, Any], FloatArrayLike],
    t_curr: ArrayLike,
    diff_ufunc: float = 1.0,
    diff_t: Any = 1e-6,
    maxiter: int = 10,
) -> tuple[ArrayLike, FloatArrayLike, int, Any]:
    """Return the root of ufunc calculated using the Newton method."""
    # implementation based on https://en.wikipedia.org/wiki/Newton%27s_method
    # strong convergence, all points below one of the two thresholds
    for k in range(maxiter):
        f_curr, payload_curr = ufunc(t_curr)

        # print(f"{f_curr / 7500}")

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        fp_curr = ufunc_prime(t_curr, payload_curr)

        t_diff = f_curr / fp_curr  # type: ignore

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

        t_curr = t_curr - t_diff  # type: ignore

    return t_curr, f_curr, k, payload_curr


def zero_doppler_plane_distance_velocity(
    dem_ecef: xr.DataArray,
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    orbit_time: xr.DataArray,
    dim: str = "axis",
) -> tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray]]:
    dem_distance = dem_ecef - orbit_interpolator.position_from_orbit_time(orbit_time)
    satellite_velocity = orbit_interpolator.velocity_from_orbit_time(orbit_time)
    plane_distance_velocity = (dem_distance * satellite_velocity).sum(dim, skipna=False)
    return plane_distance_velocity, (dem_distance, satellite_velocity)


def zero_doppler_plane_distance_velocity_prime(
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    orbit_time: xr.DataArray,
    payload: tuple[xr.DataArray, xr.DataArray],
    dim: str = "axis",
) -> xr.DataArray:
    dem_distance, satellite_velocity = payload

    plane_distance_velocity_prime = (
        dem_distance * orbit_interpolator.acceleration_from_orbit_time(orbit_time)
        - satellite_velocity**2
    ).sum(dim)
    return plane_distance_velocity_prime


def backward_geocode_simple(
    dem_ecef: xr.DataArray,
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    orbit_time_guess: xr.DataArray | float = 0.0,
    dim: str = "axis",
    zero_doppler_distance: float = 1.0,
    satellite_speed: float = 7_500.0,
    method: str = "secant",
    orbit_time_prev_shift: float = -0.1,
    maxiter: int = 10,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    diff_ufunc = zero_doppler_distance * satellite_speed

    zero_doppler = functools.partial(
        zero_doppler_plane_distance_velocity, dem_ecef, orbit_interpolator
    )

    if isinstance(orbit_time_guess, xr.DataArray):
        pass
    else:
        t_template = dem_ecef.isel({dim: 0}).drop_vars(dim).rename("azimuth_time")
        orbit_time_guess = xr.full_like(
            t_template,
            orbit_time_guess,
            dtype="float64",
        )

    if method == "secant":
        orbit_time_guess_prev = orbit_time_guess + orbit_time_prev_shift
        orbit_time, _, _, k, (dem_distance, satellite_velocity) = secant_method(
            zero_doppler,
            orbit_time_guess_prev,
            orbit_time_guess,
            diff_ufunc,
            maxiter=maxiter,
        )
    elif method in {"newton", "newton_raphson"}:
        zero_doppler_prime = functools.partial(
            zero_doppler_plane_distance_velocity_prime, orbit_interpolator
        )
        orbit_time, _, k, (dem_distance, satellite_velocity) = newton_raphson_method(
            zero_doppler,
            zero_doppler_prime,
            orbit_time_guess,
            diff_ufunc,
            maxiter=maxiter,
        )
    # print(f"iterations: {k}")
    return orbit_time, dem_distance, satellite_velocity


def backward_geocode(
    dem_ecef: xr.DataArray,
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    orbit_time_guess: xr.DataArray | float = 0.0,
    dim: str = "axis",
    zero_doppler_distance: float = 1.0,
    satellite_speed: float = 7_500.0,
    method: str = "newton",
    seed_step: tuple[int, int] | None = None,
    maxiter: int = 10,
    maxiter_after_seed: int = 1,
    orbit_time_prev_shift: float = -0.1,
) -> xr.Dataset:
    if seed_step is not None:
        dem_ecef_seed = dem_ecef.isel(
            y=slice(seed_step[0] // 2, None, seed_step[0]),
            x=slice(seed_step[1] // 2, None, seed_step[1]),
        )
        orbit_time_seed, _, _ = backward_geocode_simple(
            dem_ecef_seed,
            orbit_interpolator,
            orbit_time_guess,
            dim,
            zero_doppler_distance,
            satellite_speed,
            method,
            orbit_time_prev_shift=orbit_time_prev_shift,
        )
        orbit_time_guess = orbit_time_seed.interp_like(
            dem_ecef.sel(axis=0), kwargs={"fill_value": "extrapolate"}
        )
        maxiter = maxiter_after_seed

    orbit_time, dem_distance, satellite_velocity = backward_geocode_simple(
        dem_ecef,
        orbit_interpolator,
        orbit_time_guess,
        dim,
        zero_doppler_distance,
        satellite_speed,
        method,
        maxiter=maxiter,
        orbit_time_prev_shift=orbit_time_prev_shift,
    )

    acquisition = xr.Dataset(
        data_vars={
            "azimuth_time": orbit_interpolator.orbit_time_to_azimuth_time(orbit_time),
            "dem_distance": dem_distance,
            "satellite_velocity": satellite_velocity.transpose(*dem_distance.dims),
        }
    )
    return acquisition
