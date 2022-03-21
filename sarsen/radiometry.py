import logging
import typing as T

import numpy as np
import xarray as xr

from . import geocoding, scene

logger = logging.getLogger(__name__)

ONE_SECOND = np.timedelta64(1, "s")


def sum_weights(
    initial_weights: xr.DataArray,
    azimuth_index: xr.DataArray,
    slant_range_index: xr.DataArray,
    multilook: T.Optional[T.Tuple[int, int]] = None,
) -> xr.DataArray:
    geocoded = initial_weights.assign_coords(
        slant_range_index=slant_range_index, azimuth_index=azimuth_index
    )  # type: ignore

    stacked_geocoded = (
        geocoded.stack(z=("y", "x"))
        .reset_index("z")
        .set_index(z=("azimuth_index", "slant_range_index"))
    )

    grouped = stacked_geocoded.groupby("z")

    flat_sum = grouped.sum()

    if multilook:
        flat_sum = (
            flat_sum.unstack("z")
            .rolling(
                z_level_0=multilook[0],
                z_level_1=multilook[1],
                center=True,
                min_periods=multilook[0] * multilook[1] // 2 + 1,
            )
            .mean()
            .stack(z=("z_level_0", "z_level_1"))
        )

    stacked_sum = flat_sum.sel(z=stacked_geocoded.indexes["z"]).assign_coords(
        stacked_geocoded.coords
    )

    weights_sum: xr.DataArray = stacked_sum.set_index(z=("y", "x")).unstack("z")

    return weights_sum


def compute_gamma_area(
    dem_ecef: xr.DataArray,
    dem_direction: xr.DataArray,
) -> xr.DataArray:
    dem_oriented_area = scene.compute_dem_oriented_area(dem_ecef)
    gamma_area: xr.DataArray = xr.dot(dem_oriented_area, -dem_direction, dims="axis")  # type: ignore
    gamma_area = gamma_area.where(gamma_area > 0, 0)
    return gamma_area


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

    area = compute_gamma_area(dem_ecef, dem_coords.dem_direction)

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
    tot_area_00 = sum_weights(
        area * w_00,
        azimuth_index=azimuth_index_0,
        slant_range_index=slant_range_index_0,
    )

    logger.info("compute gamma areas 2/4")
    w_01 = abs(
        (azimuth_index_1 - azimuth_index) * (slant_range_index_0 - slant_range_index)
    )
    tot_area_01 = sum_weights(
        area * w_01,
        azimuth_index=azimuth_index_0,
        slant_range_index=slant_range_index_1,
    )

    logger.info("compute gamma areas 3/4")
    w_10 = abs(
        (azimuth_index_0 - azimuth_index) * (slant_range_index_1 - slant_range_index)
    )
    tot_area_10 = sum_weights(
        area * w_10,
        azimuth_index=azimuth_index_1,
        slant_range_index=slant_range_index_0,
    )

    logger.info("compute gamma areas 4/4")
    w_11 = abs(
        (azimuth_index_0 - azimuth_index) * (slant_range_index_0 - slant_range_index)
    )
    tot_area_11 = sum_weights(
        area * w_11,
        azimuth_index=azimuth_index_1,
        slant_range_index=slant_range_index_1,
    )

    tot_area = tot_area_00 + tot_area_01 + tot_area_10 + tot_area_11

    normalized_area = tot_area / (azimuth_spacing_m * slant_range_spacing_m)
    return normalized_area


def azimuth_slant_range_grid(
    measurement_ds: xr.DataArray,
    coordinate_conversion: T.Optional[xr.DataArray] = None,
    grouping_area_factor: T.Tuple[float, float] = (1.0, 1.0),
) -> T.Dict[str, T.Any]:

    if coordinate_conversion:
        slant_range_time0 = coordinate_conversion.slant_range_time.values[0]
        slant_range_spacing_m = (
            measurement_ds.attrs["sar:pixel_spacing_range"]
            * np.sin(measurement_ds.attrs["incidence_angle_mid_swath"])
            * grouping_area_factor[1]
        )
    else:
        slant_range_time0 = measurement_ds.slant_range_time.values[0]
        slant_range_spacing_m = (
            measurement_ds.attrs["sar:pixel_spacing_range"] * grouping_area_factor[1]
        ) * grouping_area_factor[1]

    slant_range_time_interval_s = (
        slant_range_spacing_m * 2 / geocoding.SPEED_OF_LIGHT  # ignore type
    )

    grid_parameters: T.Dict[str, T.Any] = {
        "slant_range_time0": slant_range_time0,
        "slant_range_time_interval_s": slant_range_time_interval_s,
        "slant_range_spacing_m": slant_range_spacing_m,
        "azimuth_time0": measurement_ds.azimuth_time.values[0],  # ignore type
        "azimuth_time_interval_s": measurement_ds.attrs["azimuth_time_interval"]
        * grouping_area_factor[0],
        "azimuth_spacing_m": measurement_ds.attrs["sar:pixel_spacing_azimuth"]
        * grouping_area_factor[0],
    }
    return grid_parameters
