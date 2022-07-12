import logging
from typing import Any, Dict, Optional, Tuple

import flox.xarray
import numpy as np
import xarray as xr

from . import geocoding, scene

logger = logging.getLogger(__name__)

ONE_SECOND = np.timedelta64(1, "s")


def sum_weights1(
    initial_weights: xr.DataArray,
    azimuth_index: xr.DataArray,
    slant_range_index: xr.DataArray,
    multilook: Optional[Tuple[int, int]] = None,
) -> xr.DataArray:
    geocoded = initial_weights.assign_coords(
        slant_range_index=slant_range_index, azimuth_index=azimuth_index
    )

    flat_sum: xr.DataArray = flox.xarray.xarray_reduce(
        geocoded,
        geocoded.slant_range_index,
        geocoded.azimuth_index,
        func="sum",
        method="map-reduce",
    )

    if multilook:
        flat_sum = flat_sum.rolling(
            azimuth_index=multilook[0],
            slant_range_index=multilook[1],
            center=True,
            min_periods=multilook[0] * multilook[1] // 2 + 1,
        ).mean()

    weights_sum = flat_sum.interp(
        slant_range_index=slant_range_index,
        azimuth_index=azimuth_index,
        method="nearest",
    )

    return weights_sum


def compute_gamma_area(
    dem_ecef: xr.DataArray,
    dem_direction: xr.DataArray,
) -> xr.DataArray:
    dem_oriented_area = scene.compute_dem_oriented_area(dem_ecef)
    gamma_area: xr.DataArray = xr.dot(dem_oriented_area, -dem_direction, dims="axis")  # type: ignore
    gamma_area = gamma_area.where(gamma_area > 0, 0)
    return gamma_area


def sum_weights(
    initial_weights: xr.DataArray,
    input_indices: Dict[str, xr.DataArray] = {},
    output_indices: Dict[str, xr.DataArray] = {},
    multilook: Optional[Dict[str, int]] = None,
) -> xr.DataArray:
    initial_weights = initial_weights.assign_coords(input_indices)

    flat_sum: xr.DataArray = flox.xarray.xarray_reduce(
        initial_weights,
        *input_indices.values(),
        func="sum",
        method="map-reduce",
    )

    if multilook:
        flat_sum = flat_sum.rolling(
            multilook,
            center=True,
            min_periods=np.prod(list(multilook.values())) // 2 + 1,
        ).mean()
    weights_sum = flat_sum.interp(output_indices, method="nearest")
    return weights_sum


def gamma_weights_nearest(
    acquisition: xr.Dataset,
    slant_range_time0: float,
    azimuth_time0: np.datetime64,
    slant_range_time_interval_s: float,
    azimuth_time_interval_s: float,
    slant_range_spacing_m: int = 1,
    azimuth_spacing_m: int = 1,
    oversampling: Tuple[int, int] = (1, 1),
) -> xr.DataArray:

    acquisition = acquisition.copy()
    acquisition["azimuth_time"] = (
        acquisition.azimuth_time - azimuth_time0
    ) / ONE_SECOND
    out_azimuth_index = np.round(
        acquisition.azimuth_time / azimuth_time_interval_s
    ).astype(int)
    out_slant_range_index = np.round(
        (acquisition.slant_range_time - slant_range_time0) / slant_range_time_interval_s
    ).astype(int)

    if oversampling[0] != 1 or oversampling[1] != 1:
        acquisition = acquisition.interp(
            x=np.linspace(
                acquisition.x.max(),
                acquisition.x.min(),
                acquisition.x.size * oversampling[0],
            ),
            y=np.linspace(
                acquisition.y.max(),
                acquisition.y.min(),
                acquisition.y.size * oversampling[1],
            ),
        ).persist()
        azimuth_index = np.round(
            acquisition.azimuth_time / azimuth_time_interval_s
        ).astype(int)
        slant_range_index = np.round(
            (acquisition.slant_range_time - slant_range_time0)
            / slant_range_time_interval_s
        ).astype(int)
    else:
        azimuth_index = out_azimuth_index
        slant_range_index = out_slant_range_index

    tot_area = sum_weights(
        acquisition["gamma_area"],
        input_indices={
            "azimuth_time": azimuth_index,
            "slant_range_time": slant_range_index,
        },
        output_indices={
            "azimuth_time": out_azimuth_index,
            "slant_range_time": out_slant_range_index,
        },
    )

    return tot_area / (
        azimuth_spacing_m * slant_range_spacing_m * oversampling[0] * oversampling[1]
    )


def gamma_weights_bilinear(
    acquisition: xr.Dataset,
    slant_range_time0: float,
    azimuth_time0: np.datetime64,
    slant_range_time_interval_s: float,
    azimuth_time_interval_s: float,
    slant_range_spacing_m: float = 1,
    azimuth_spacing_m: float = 1,
    oversampling: Tuple[int, int] = (1, 1),
) -> xr.DataArray:
    acquisition = acquisition.copy()
    acquisition["azimuth_time"] = (
        acquisition.azimuth_time - azimuth_time0
    ) / ONE_SECOND
    out_azimuth_index = acquisition.azimuth_time / azimuth_time_interval_s
    out_slant_range_index = (
        acquisition.slant_range_time - slant_range_time0
    ) / slant_range_time_interval_s

    if oversampling[0] != 1 or oversampling[1] != 1:
        acquisition = acquisition.interp(
            x=np.linspace(
                acquisition.x.max(),
                acquisition.x.min(),
                acquisition.x.size * oversampling[0],
            ),
            y=np.linspace(
                acquisition.y.max(),
                acquisition.y.min(),
                acquisition.y.size * oversampling[1],
            ),
        )
        acquisition = acquisition.persist()
        azimuth_index = acquisition.azimuth_time / azimuth_time_interval_s
        slant_range_index = (
            acquisition.slant_range_time - slant_range_time0
        ) / slant_range_time_interval_s
    else:
        azimuth_index = out_azimuth_index
        slant_range_index = out_slant_range_index

    out_azimuth_index = np.round(out_azimuth_index).astype(int)
    out_slant_range_index = np.round(out_slant_range_index).astype(int)
    # compute dem image coordinates

    slant_range_index_0 = np.floor(slant_range_index).astype(int).compute()
    slant_range_index_1 = np.ceil(slant_range_index).astype(int).compute()
    azimuth_index_0 = np.floor(azimuth_index).astype(int).compute()
    azimuth_index_1 = np.ceil(azimuth_index).astype(int).compute()

    logger.info("compute gamma areas 1/4")
    w_00 = abs(
        (azimuth_index_1 - azimuth_index) * (slant_range_index_1 - slant_range_index)
    )
    tot_area_00 = sum_weights(
        acquisition["gamma_area"] * w_00,
        input_indices={
            "azimuth_time": azimuth_index_0,
            "slant_range_time": slant_range_index_0,
        },
        output_indices={
            "azimuth_time": out_azimuth_index,
            "slant_range_time": out_slant_range_index,
        },
    )

    logger.info("compute gamma areas 2/4")
    w_01 = abs(
        (azimuth_index_1 - azimuth_index) * (slant_range_index_0 - slant_range_index)
    )
    tot_area_01 = sum_weights(
        acquisition["gamma_area"] * w_01,
        input_indices={
            "azimuth_time": azimuth_index_0,
            "slant_range_time": slant_range_index_1,
        },
        output_indices={
            "azimuth_time": out_azimuth_index,
            "slant_range_time": out_slant_range_index,
        },
    )

    logger.info("compute gamma areas 3/4")
    w_10 = abs(
        (azimuth_index_0 - azimuth_index) * (slant_range_index_1 - slant_range_index)
    )
    tot_area_10 = sum_weights(
        acquisition["gamma_area"] * w_10,
        input_indices={
            "azimuth_time": azimuth_index_1,
            "slant_range_time": slant_range_index_0,
        },
        output_indices={
            "azimuth_time": out_azimuth_index,
            "slant_range_time": out_slant_range_index,
        },
    )

    logger.info("compute gamma areas 4/4")
    w_11 = abs(
        (azimuth_index_0 - azimuth_index) * (slant_range_index_0 - slant_range_index)
    )
    tot_area_11 = sum_weights(
        acquisition["gamma_area"] * w_11,
        input_indices={
            "azimuth_time": azimuth_index_1,
            "slant_range_time": slant_range_index_1,
        },
        output_indices={
            "azimuth_time": out_azimuth_index,
            "slant_range_time": out_slant_range_index,
        },
    )

    tot_area = tot_area_00 + tot_area_01 + tot_area_10 + tot_area_11

    # acquisition["azimuth_time"] = acquisition["azimuth_time_float"] * ONE_SECOND + azimuth_time0

    return tot_area / (
        azimuth_spacing_m * slant_range_spacing_m * oversampling[0] * oversampling[1]
    )


def azimuth_slant_range_grid(
    attrs: Dict[str, Any],
    slant_range_time0: float,
    azimuth_time0: float,
    grouping_area_factor: Tuple[float, float] = (3.0, 3.0),
) -> Dict[str, Any]:

    if attrs["product_type"] == "SLC":
        slant_range_spacing_m = (
            attrs["range_pixel_spacing"]
            * np.sin(attrs["incidence_angle_mid_swath"])
            * grouping_area_factor[1]
        )
    else:
        slant_range_spacing_m = attrs["range_pixel_spacing"] * grouping_area_factor[1]

    slant_range_time_interval_s = (
        slant_range_spacing_m * 2 / geocoding.SPEED_OF_LIGHT  # ignore type
    )

    grid_parameters: Dict[str, Any] = {
        "slant_range_time0": slant_range_time0,
        "slant_range_time_interval_s": slant_range_time_interval_s,
        "slant_range_spacing_m": slant_range_spacing_m,
        "azimuth_time0": azimuth_time0,
        "azimuth_time_interval_s": attrs["azimuth_time_interval"]
        * grouping_area_factor[0],
        "azimuth_spacing_m": attrs["azimuth_pixel_spacing"] * grouping_area_factor[0],
    }
    return grid_parameters
