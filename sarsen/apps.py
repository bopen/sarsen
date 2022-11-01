import logging
from typing import Any, Callable, Dict, Optional, Tuple
from unittest import mock

import numpy as np
import rioxarray
import xarray as xr

from . import chunking, datamodel, geocoding, orbit, radiometry, scene

logger = logging.getLogger(__name__)


def simulate_acquisition(
    dem_ecef: xr.DataArray,
    position_ecef: xr.DataArray,
    slant_range_time_to_ground_range: Callable[
        [xr.DataArray, xr.DataArray], xr.DataArray
    ],
    correct_radiometry: Optional[str] = None,
) -> xr.Dataset:
    """Compute the image coordinates of the DEM given the satellite orbit."""

    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(position_ecef)
    position_ecef = orbit_interpolator.position()
    velocity_ecef = orbit_interpolator.velocity()

    acquisition = geocoding.backward_geocode(dem_ecef, position_ecef, velocity_ecef)

    slant_range = (acquisition.dem_distance**2).sum(dim="axis") ** 0.5
    slant_range_time = 2.0 / geocoding.SPEED_OF_LIGHT * slant_range

    acquisition["slant_range_time"] = slant_range_time

    maybe_ground_range = slant_range_time_to_ground_range(
        acquisition.azimuth_time,
        slant_range_time,
    )
    if maybe_ground_range is not None:
        acquisition["ground_range"] = maybe_ground_range.drop_vars("azimuth_time")
    if correct_radiometry is not None:
        gamma_area = radiometry.compute_gamma_area(
            dem_ecef, acquisition.dem_distance / slant_range
        )
        acquisition["gamma_area"] = gamma_area

    acquisition = acquisition.drop_vars(["dem_distance", "satellite_direction", "axis"])

    return acquisition


def terrain_correction(
    product: datamodel.SarProduct,
    dem_urlpath: str,
    output_urlpath: Optional[str] = "GTC.tif",
    simulated_urlpath: Optional[str] = None,
    correct_radiometry: Optional[str] = None,
    interp_method: xr.core.types.InterpOptions = "nearest",
    grouping_area_factor: Tuple[float, float] = (3.0, 3.0),
    open_dem_raster_kwargs: Dict[str, Any] = {},
    chunks: Optional[int] = 1024,
    radiometry_chunks: int = 2048,
    radiometry_bound: int = 128,
    enable_dask_distributed: bool = False,
    client_kwargs: Dict[str, Any] = {"processes": False},
) -> xr.DataArray:
    """Apply the terrain-correction to sentinel-1 SLC and GRD products.

    :param product: SarProduct instance representing the input data
    :param dem_urlpath: dem path or url
    :param orbit_group: overrides the orbit group name
    :param calibration_group: overrides the calibration group name
    :param coordinate_conversion_group: overrides the coordinate_conversion group name
    :param output_urlpath: output path or url
    :param correct_radiometry: default `None`. If `correct_radiometry=None`the radiometric terrain
    correction is not applied. `correct_radiometry=gamma_bilinear` applies the gamma flattening classic
    algorithm using bilinear interpolation to compute the weights. `correct_radiometry=gamma_nearest`
    applies the gamma flattening using nearest neighbours instead of bilinear interpolation.
    'gamma_nearest' significantly reduces the processing time
    :param interp_method: interpolation method for product resampling.
    The interpolation methods are the methods supported by ``xarray.DataArray.interp``
    :param multilook: multilook factor. If `None` the multilook is not applied
    :param grouping_area_factor: is a tuple of floats greater than 1. The default is `(1, 1)`.
    The `grouping_area_factor`  can be increased (i) to speed up the processing or
    (ii) when the input DEM resolution is low.
    The Gamma Flattening usually works properly if the pixel size of the input DEM is much smaller
    than the pixel size of the input Sentinel-1 product. Otherwise, the output may have radiometric distortions.
    This problem can be avoided by increasing the `grouping_area_factor`.
    Be aware that `grouping_area_factor` too high may degrade the final result
    :param open_dem_raster_kwargs: additional keyword arguments passed on to ``xarray.open_dataset``
    to open the `dem_urlpath`
    """
    # rioxarray must be imported explicitly or accesses to `.rio` may fail in dask
    assert rioxarray.__version__

    allowed_correct_radiometry = [None, "gamma_bilinear", "gamma_nearest"]
    if correct_radiometry not in allowed_correct_radiometry:
        raise ValueError(
            f"{correct_radiometry=}. Must be one of: {allowed_correct_radiometry}"
        )
    if output_urlpath is None and simulated_urlpath is None:
        raise ValueError("No output selected")

    output_chunks = chunks if chunks is not None else 512

    to_raster_kwargs: Dict[str, Any] = {}
    if enable_dask_distributed:
        from dask.distributed import Client, Lock

        client = Client(**client_kwargs)  # type: ignore
        to_raster_kwargs["lock"] = Lock("rio", client=client)  # type: ignore
        to_raster_kwargs["compute"] = False
        print(f"Dask distributed dashboard at: {client.dashboard_link}")

    logger.info(f"open DEM {dem_urlpath!r}")

    dem_raster = scene.open_dem_raster(
        dem_urlpath, chunks=chunks, **open_dem_raster_kwargs
    )

    allowed_product_types = ["GRD", "SLC"]
    if product.product_type not in allowed_product_types:
        raise ValueError(
            f"{product.product_type=}. Must be one of: {allowed_product_types}"
        )

    logger.info("pre-process DEM")

    dem_ecef = xr.map_blocks(
        scene.convert_to_dem_ecef, dem_raster, kwargs={"source_crs": dem_raster.rio.crs}
    )
    dem_ecef = dem_ecef.drop_vars(dem_ecef.rio.grid_mapping)

    logger.info("simulate acquisition")

    template_raster = dem_raster.drop_vars(dem_raster.rio.grid_mapping) * 0.0
    acquisition_template = xr.Dataset(
        data_vars={
            "slant_range_time": template_raster,
            "azimuth_time": template_raster.astype("datetime64[ns]"),
        }
    )
    if product.product_type == "GRD":
        acquisition_template["ground_range"] = template_raster
    if correct_radiometry is not None:
        acquisition_template["gamma_area"] = template_raster

    acquisition = xr.map_blocks(
        simulate_acquisition,
        dem_ecef,
        kwargs={
            "position_ecef": product.state_vectors(),
            "slant_range_time_to_ground_range": product.slant_range_time_to_ground_range,
            "correct_radiometry": correct_radiometry,
        },
        template=acquisition_template,
    )

    if correct_radiometry is not None:
        logger.info("simulate radiometry")

        grid_parameters = product.grid_parameters(grouping_area_factor)

        if correct_radiometry == "gamma_bilinear":
            gamma_weights = radiometry.gamma_weights_bilinear
        elif correct_radiometry == "gamma_nearest":
            gamma_weights = radiometry.gamma_weights_nearest

        acquisition = acquisition.persist()

        simulated_beta_nought = chunking.map_ovelap(
            obj=acquisition,
            function=gamma_weights,
            chunks=radiometry_chunks,
            bound=radiometry_bound,
            kwargs=grid_parameters,
            template=template_raster,
        )
        simulated_beta_nought.x.attrs.update(dem_raster.x.attrs)
        simulated_beta_nought.y.attrs.update(dem_raster.y.attrs)
        simulated_beta_nought.rio.set_crs(dem_raster.rio.crs)

    if simulated_urlpath is not None:
        if output_urlpath is not None:
            simulated_beta_nought.persist()

        logger.info("save simulated")

        maybe_delayed = simulated_beta_nought.rio.to_raster(
            simulated_urlpath,
            dtype=np.float32,
            tiled=True,
            blockxsize=output_chunks,
            blockysize=output_chunks,
            compress="ZSTD",
            num_threads="ALL_CPUS",
            **to_raster_kwargs,
        )

        if enable_dask_distributed:
            maybe_delayed.compute()

    if output_urlpath is None:
        return simulated_beta_nought

    logger.info("calibrate image")

    beta_nought = product.beta_nought()

    logger.info("terrain-correct image")

    if product.product_type == "GRD":
        interp_kwargs = {"ground_range": acquisition.ground_range}
    else:
        interp_kwargs = {"slant_range_time": acquisition.slant_range_time}

    # HACK: we monkey-patch away an optimisation in xr.DataArray.interp that actually makes
    #   the interpolation much slower when indeces are dask arrays.
    with mock.patch("xarray.core.missing._localize", lambda o, i: (o, i)):
        geocoded = beta_nought.interp(
            method=interp_method,
            azimuth_time=acquisition.azimuth_time,
            **interp_kwargs,
        )

    if correct_radiometry is not None:
        geocoded = geocoded / simulated_beta_nought

    geocoded.attrs.update(beta_nought.attrs)
    geocoded.x.attrs.update(dem_raster.x.attrs)
    geocoded.y.attrs.update(dem_raster.y.attrs)
    geocoded.rio.set_crs(dem_raster.rio.crs)

    logger.info("save output")

    maybe_delayed = geocoded.rio.to_raster(
        output_urlpath,
        dtype=np.float32,
        tiled=True,
        blockxsize=output_chunks,
        blockysize=output_chunks,
        compress="ZSTD",
        num_threads="ALL_CPUS",
        **to_raster_kwargs,
    )

    if enable_dask_distributed:
        maybe_delayed.compute()

    return geocoded
