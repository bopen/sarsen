import logging
from typing import Any, Container
from unittest import mock

import numpy as np
import rioxarray
import xarray as xr

from . import chunking, datamodel, geocoding, orbit, radiometry, scene

logger = logging.getLogger(__name__)


SPEED_OF_LIGHT = 299_792_458.0  # m / s


def make_simulate_acquisition_template(
    template_raster: xr.DataArray,
    correct_radiometry: str | None = None,
) -> xr.Dataset:
    acquisition_template = xr.Dataset(
        data_vars={
            "slant_range_time": template_raster,
            "azimuth_time": template_raster.astype("datetime64[ns]"),
        }
    )
    include_variables = {"slant_range_time", "azimuth_time"}
    if correct_radiometry is not None:
        acquisition_template["gamma_area"] = template_raster
        include_variables.add("gamma_area")

    return acquisition_template


def simulate_acquisition(
    dem_ecef: xr.DataArray,
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    include_variables: Container[str] = (),
    azimuth_time: xr.DataArray | float = 0.0,
    **kwargs: Any,
) -> xr.Dataset:
    """Compute the image coordinates of the DEM given the satellite orbit."""
    acquisition = geocoding.backward_geocode(
        dem_ecef, orbit_interpolator, azimuth_time, **kwargs
    )

    slant_range = (acquisition.dem_distance**2).sum(dim="axis") ** 0.5
    slant_range_time = 2.0 / SPEED_OF_LIGHT * slant_range

    acquisition["slant_range_time"] = slant_range_time

    if include_variables and "gamma_area" in include_variables:
        gamma_area = radiometry.compute_gamma_area(
            dem_ecef, acquisition.dem_distance / slant_range
        )
        acquisition["gamma_area"] = gamma_area

    for data_var_name in acquisition.data_vars:
        if include_variables and data_var_name not in include_variables:
            acquisition = acquisition.drop_vars(data_var_name)  # type: ignore

    # drop coordinates that are not associated with any data variable
    for coord_name in acquisition.coords:
        if all(coord_name not in dv.coords for dv in acquisition.data_vars.values()):
            acquisition = acquisition.drop_vars(coord_name)  # type: ignore

    return acquisition


def map_simulate_acquisition(
    dem_ecef: xr.DataArray,
    orbit_interpolator: orbit.OrbitPolyfitInterpolator,
    template_raster: xr.DataArray | None = None,
    correct_radiometry: str | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    if template_raster is None:
        template_raster = dem_ecef.isel(axis=0).drop_vars(["axis", "spatial_ref"]) * 0.0
    acquisition_template = make_simulate_acquisition_template(
        template_raster, correct_radiometry
    )
    acquisition = xr.map_blocks(
        simulate_acquisition,
        dem_ecef.drop_vars("spatial_ref"),
        kwargs={
            "orbit_interpolator": orbit_interpolator,
            "include_variables": list(acquisition_template.data_vars),
        }
        | kwargs,
        template=acquisition_template,
    )
    return acquisition


def do_terrain_correction(
    product: datamodel.SarProduct,
    dem_raster: xr.DataArray,
    convert_to_dem_ecef_kwargs: dict[str, Any] = {},
    correct_radiometry: str | None = None,
    interp_method: xr.core.types.InterpOptions = "nearest",
    grouping_area_factor: tuple[float, float] = (3.0, 3.0),
    radiometry_chunks: int = 2048,
    radiometry_bound: int = 128,
    seed_step: tuple[int, int] | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    logger.info("pre-process DEM")

    dem_ecef = xr.map_blocks(
        scene.convert_to_dem_ecef, dem_raster, kwargs=convert_to_dem_ecef_kwargs
    )

    logger.info("simulate acquisition")

    template_raster = dem_ecef.isel(axis=0).drop_vars(["axis", "spatial_ref"]) * 0.0

    orbit_interpolator = orbit.OrbitPolyfitInterpolator.from_position(
        product.state_vectors()
    )

    acquisition = map_simulate_acquisition(
        dem_ecef,
        orbit_interpolator,
        correct_radiometry=correct_radiometry,
        seed_step=seed_step,
    )

    simulated_beta_nought = None
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
        simulated_beta_nought.attrs["long_name"] = "terrain-simulated beta nought"

        simulated_beta_nought.x.attrs.update(dem_ecef.x.attrs)
        simulated_beta_nought.y.attrs.update(dem_ecef.y.attrs)
        simulated_beta_nought.rio.write_crs(dem_ecef.rio.crs, inplace=True)

    logger.info("calibrate image")

    beta_nought = product.beta_nought()

    logger.info("terrain-correct image")

    # HACK: we monkey-patch away an optimisation in xr.DataArray.interp that actually makes
    #   the interpolation much slower when indeces are dask arrays.
    with mock.patch("xarray.core.missing._localize", lambda o, i: (o, i)):
        geocoded = product.interp_sar(
            beta_nought,
            azimuth_time=acquisition.azimuth_time,
            slant_range_time=acquisition.slant_range_time,
            method=interp_method,
        )

    if correct_radiometry is not None:
        assert simulated_beta_nought is not None
        geocoded = geocoded / simulated_beta_nought
        geocoded.attrs["long_name"] = "terrain-corrected gamma nought"

    geocoded.x.attrs.update(dem_ecef.x.attrs)
    geocoded.y.attrs.update(dem_ecef.y.attrs)
    geocoded.rio.write_crs(dem_ecef.rio.crs, inplace=True)

    return geocoded, simulated_beta_nought


def terrain_correction(
    product: datamodel.SarProduct,
    dem_urlpath: str,
    output_urlpath: str | None = "GTC.tif",
    simulated_urlpath: str | None = None,
    correct_radiometry: str | None = None,
    interp_method: xr.core.types.InterpOptions = "nearest",
    grouping_area_factor: tuple[float, float] = (3.0, 3.0),
    open_dem_raster_kwargs: dict[str, Any] = {},
    chunks: int | None = 1024,
    radiometry_chunks: int = 2048,
    radiometry_bound: int = 128,
    enable_dask_distributed: bool = False,
    client_kwargs: dict[str, Any] = {"processes": False},
    seed_step: tuple[int, int] | None = None,
) -> xr.DataArray:
    """Apply the terrain-correction to sentinel-1 SLC and GRD products.

    :param product: SarProduct instance representing the input data
    :param dem_urlpath: dem path or url
    :param output_urlpath: output path or url
    :param correct_radiometry: default `None`. If `correct_radiometry=None`the radiometric terrain
    correction is not applied. `correct_radiometry=gamma_bilinear` applies the gamma flattening classic
    algorithm using bilinear interpolation to compute the weights. `correct_radiometry=gamma_nearest`
    applies the gamma flattening using nearest neighbours instead of bilinear interpolation.
    'gamma_nearest' significantly reduces the processing time
    :param interp_method: interpolation method for product resampling.
    The interpolation methods are the methods supported by ``xarray.DataArray.interp``
    :param grouping_area_factor: is a tuple of floats greater than 1. The default is `(1, 1)`.
    The `grouping_area_factor`  can be increased (i) to speed up the processing or
    (ii) when the input DEM resolution is low.
    The Gamma Flattening usually works properly if the pixel size of the input DEM is much smaller
    than the pixel size of the input Sentinel-1 product.
    Otherwise, the output may have radiometric distortions.
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
    if simulated_urlpath is not None and correct_radiometry is None:
        raise ValueError("Simulation cannot be saved")
    if output_urlpath is None and simulated_urlpath is None:
        raise ValueError("No output selected")

    allowed_product_types = ["GRD", "SLC"]
    if product.product_type not in allowed_product_types:
        raise ValueError(
            f"{product.product_type=}. Must be one of: {allowed_product_types}"
        )

    output_chunks = chunks if chunks is not None else 512

    to_raster_kwargs: dict[str, Any] = {}
    if enable_dask_distributed:
        from dask.distributed import Client, Lock

        client = Client(**client_kwargs)
        to_raster_kwargs["lock"] = Lock("rio", client=client)
        to_raster_kwargs["compute"] = False
        print(f"Dask distributed dashboard at: {client.dashboard_link}")

    logger.info(f"open DEM {dem_urlpath!r}")

    dem_raster = scene.open_dem_raster(
        dem_urlpath, chunks=chunks, **open_dem_raster_kwargs
    )

    geocoded, simulated_beta_nought = do_terrain_correction(
        product=product,
        dem_raster=dem_raster,
        correct_radiometry=correct_radiometry,
        interp_method=interp_method,
        grouping_area_factor=grouping_area_factor,
        radiometry_chunks=radiometry_chunks,
        radiometry_bound=radiometry_bound,
        seed_step=seed_step,
    )

    if simulated_urlpath is not None:
        assert simulated_beta_nought is not None
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
        assert simulated_beta_nought is not None
        return simulated_beta_nought

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
