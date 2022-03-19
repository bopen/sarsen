"""
Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019
https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""
import functools
import logging
import typing as T

import numpy as np
import numpy.typing as npt
import xarray as xr

logger = logging.getLogger(__name__)


SPEED_OF_LIGHT = 299_792_458.0  # m / s
ONE_SECOND = np.timedelta64(1, "s")

TimedeltaArrayLike = T.TypeVar("TimedeltaArrayLike", bound=npt.ArrayLike)
FloatArrayLike = T.TypeVar("FloatArrayLike", bound=npt.ArrayLike)


def secant_method(
    ufunc: T.Callable[[TimedeltaArrayLike], T.Tuple[FloatArrayLike, FloatArrayLike]],
    t_prev: TimedeltaArrayLike,
    t_curr: TimedeltaArrayLike,
    diff_ufunc: float = 1.0,
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
    position_ecef: xr.DataArray,
    velocity_ecef: xr.DataArray,
    azimuth_time: T.Optional[xr.DataArray] = None,
    dim: str = "axis",
    diff_ufunc: float = 1.0,
) -> xr.Dataset:
    direction_ecef = (
        velocity_ecef / xr.dot(velocity_ecef, velocity_ecef, dims=dim) ** 0.5  # type: ignore
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
    _, _, _, dem_distance = secant_method(zero_doppler, t_prev, t_curr, diff_ufunc)
    dem_slant_range = xr.dot(dem_distance, dem_distance, dims=dim) ** 0.5  # type: ignore
    slant_range_time = 2.0 / SPEED_OF_LIGHT * dem_slant_range
    dem_direction = dem_distance / dem_slant_range
    simulation = xr.merge(
        [
            slant_range_time.rename("slant_range_time"),
            dem_direction.rename("dem_direction"),
        ]
    )
    return simulation.reset_coords("azimuth_time")


def dem_area_gamma(
    dem_ecef: xr.DataArray,
    dem_direction: xr.DataArray,
) -> xr.DataArray:

    x_corners: npt.ArrayLike = np.concatenate(
        [
            [dem_ecef.x[0] + (dem_ecef.x[0] - dem_ecef.x[1]) / 2],
            ((dem_ecef.x.shift(x=-1) + dem_ecef.x) / 2)[:-1].data,
            [dem_ecef.x[-1] + (dem_ecef.x[-1] - dem_ecef.x[-2]) / 2],
        ]
    )
    y_corners: npt.ArrayLike = np.concatenate(
        [
            [dem_ecef.y[0] + (dem_ecef.y[0] - dem_ecef.y[1]) / 2],
            ((dem_ecef.y.shift(y=-1) + dem_ecef.y) / 2)[:-1].data,
            [dem_ecef.y[-1] + (dem_ecef.y[-1] - dem_ecef.y[-2]) / 2],
        ]
    )
    dem_ecef_corners = dem_ecef.interp(
        {"x": x_corners, "y": y_corners},
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    dx = dem_ecef_corners.diff("x", 1)
    dy = dem_ecef_corners.diff("y", 1)

    dx1 = dx.isel(y=slice(1, None)).assign_coords(dem_ecef.coords)  # type: ignore
    dy1 = dy.isel(x=slice(1, None)).assign_coords(dem_ecef.coords)  # type: ignore
    dx2 = dx.isel(y=slice(None, -1)).assign_coords(dem_ecef.coords)  # type: ignore
    dy2 = dy.isel(x=slice(None, -1)).assign_coords(dem_ecef.coords)  # type: ignore

    cross_1 = xr.cross(dx1, dy1, dim="axis") / 2
    sign = np.sign(
        xr.dot(cross_1, dem_ecef, dims="axis")  # type: ignore
    )  # ensure direction out of DEM
    area_t1 = xr.dot(sign * cross_1, -dem_direction, dims="axis")  # type: ignore
    area_t1 = area_t1.where(area_t1 > 0, 0)

    cross_2 = xr.cross(dx2, dy2, dim="axis") / 2
    sign = np.sign(
        xr.dot(cross_2, dem_ecef, dims="axis")  # type: ignore
    )  # ensure direction out of DEM
    area_t2 = xr.dot(sign * cross_2, -dem_direction, dims="axis")  # type: ignore
    area_t2 = area_t2.where(area_t2 > 0, 0)

    area: xr.DataArray = area_t1 + area_t2

    return area


def sum_area_on_image_pixels(
    dem_area: xr.DataArray,
    azimuth_index: xr.DataArray,
    slant_range_index: xr.DataArray,
) -> xr.DataArray:
    dem_area = dem_area.assign_coords(  # type: ignore
        {
            "azimuth_index": azimuth_index,
            "slant_range_index": slant_range_index,
        }
    )

    dem_area = dem_area.stack(z=("x", "y")).reset_index("z")
    dem_area = dem_area.set_index(z=("azimuth_index", "slant_range_index"))

    sum_area: xr.DataArray = dem_area.groupby("z").sum()

    tot_area = sum_area.sel(z=dem_area.indexes["z"])
    tot_area = tot_area.assign_coords(dem_area.coords)  # type: ignore
    tot_area = tot_area.reset_index("z").set_index(z=("x", "y")).unstack("z")
    return tot_area


def gamma_weights(
    dem_ecef: xr.DataArray,
    dem_coords: xr.Dataset,
    slant_range_time0: float,
    azimuth_time0: np.datetime64,
    slant_range_time_interval_s: float,
    azimuth_time_interval_s: float,
    slant_range_spacing_m: float = 1,
    azimuth_spacing_m: float = 1,
) -> xr.DataArray:

    area = dem_area_gamma(dem_ecef, dem_coords.dem_direction)

    # compute dem image coordinates
    azimuth_index = ((dem_coords.azimuth_time - azimuth_time0) / ONE_SECOND) / (
        azimuth_time_interval_s
    )

    slant_range_index = (dem_coords.slant_range_time - slant_range_time0) / (
        slant_range_time_interval_s
    )

    slant_range_index_0 = np.floor(slant_range_index).astype(int)
    slant_range_index_1 = np.ceil(slant_range_index).astype(int)
    azimuth_index_0 = np.floor(azimuth_index).astype(int)
    azimuth_index_1 = np.ceil(azimuth_index).astype(int)

    logger.info("compute gamma areas 1/4")
    w_00 = abs(
        (azimuth_index_1 - azimuth_index) * (slant_range_index_1 - slant_range_index)
    )
    tot_area_00 = sum_area_on_image_pixels(
        area * w_00,
        azimuth_index=azimuth_index_0,
        slant_range_index=slant_range_index_0,
    )

    logger.info("compute gamma areas 2/4")
    w_01 = abs(
        (azimuth_index_1 - azimuth_index) * (slant_range_index_0 - slant_range_index)
    )
    tot_area_01 = sum_area_on_image_pixels(
        area * w_01,
        azimuth_index=azimuth_index_0,
        slant_range_index=slant_range_index_1,
    )

    logger.info("compute gamma areas 3/4")
    w_10 = abs(
        (azimuth_index_0 - azimuth_index) * (slant_range_index_1 - slant_range_index)
    )
    tot_area_10 = sum_area_on_image_pixels(
        area * w_10,
        azimuth_index=azimuth_index_1,
        slant_range_index=slant_range_index_0,
    )

    logger.info("compute gamma areas 4/4")
    w_11 = abs(
        (azimuth_index_0 - azimuth_index) * (slant_range_index_0 - slant_range_index)
    )
    tot_area_11 = sum_area_on_image_pixels(
        area * w_11,
        azimuth_index=azimuth_index_1,
        slant_range_index=slant_range_index_1,
    )

    tot_area = tot_area_00 + tot_area_01 + tot_area_10 + tot_area_11

    normalized_area = tot_area / (azimuth_spacing_m * slant_range_spacing_m)
    return normalized_area
