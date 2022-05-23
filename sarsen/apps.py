import logging
import typing as T

import numpy as np
import rioxarray
import xarray as xr
import xarray_sentinel

from . import geocoding, orbit, radiometry, scene

logger = logging.getLogger(__name__)


def open_dataset_autodetect(
    product_urlpath: str,
    group: T.Optional[str] = None,
    chunks: T.Optional[T.Union[int, T.Dict[str, int]]] = None,
    **kwargs: T.Any,
) -> T.Tuple[xr.Dataset, T.Dict[str, T.Any]]:
    try:
        ds = xr.open_dataset(
            product_urlpath, engine="sentinel-1", group=group, chunks=chunks, **kwargs  # type: ignore
        )
    except FileNotFoundError:
        # re-try with Planetary Computer option
        kwargs[
            "override_product_files"
        ] = "{dirname}/{prefix}{swath}-{polarization}{ext}"
        ds = xr.open_dataset(
            product_urlpath, engine="sentinel-1", group=group, chunks=chunks, **kwargs  # type: ignore
        )
    return ds, kwargs


def product_info(
    product_urlpath: str,
    **kwargs: T.Any,
) -> T.Dict[str, T.Any]:
    """Get information about the Sentinel-1 product."""
    root_ds = xr.open_dataset(product_urlpath, engine="sentinel-1", **kwargs)  # type: ignore

    measurement_groups = [g for g in root_ds.attrs["subgroups"] if g.count("/") == 1]

    gcp_group = measurement_groups[0] + "/gcp"

    gcp, kwargs = open_dataset_autodetect(product_urlpath, group=gcp_group, **kwargs)

    bbox = [
        gcp.attrs["geospatial_lon_min"],
        gcp.attrs["geospatial_lat_min"],
        gcp.attrs["geospatial_lon_max"],
        gcp.attrs["geospatial_lat_max"],
    ]

    product_info = {
        "product_type": root_ds.attrs["product_type"],
        "mode": root_ds.attrs["mode"],
        "swaths": root_ds.attrs["swaths"],
        "transmitter_receiver_polarisations": root_ds.attrs[
            "transmitter_receiver_polarisations"
        ],
        "measurement_groups": measurement_groups,
        "bbox": bbox,
    }

    return product_info


def simulate_acquisition(
    dem_ecef: xr.DataArray,
    position_ecef: xr.DataArray,
) -> xr.Dataset:
    """Compute the image coordinates of the DEM given the satellite orbit."""

    logger.info("interpolate orbit")

    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(position_ecef)
    position_ecef = orbit_interpolator.position()
    velocity_ecef = orbit_interpolator.velocity()

    logger.info("geocode")

    acquisition = geocoding.backward_geocode(dem_ecef, position_ecef, velocity_ecef)

    return acquisition


def interpolate_measurement(
    azimuth_time: xr.DataArray,
    range: xr.DataArray,
    image: xr.DataArray,
    multilook: T.Optional[T.Tuple[int, int]] = None,
    interp_method: str = "nearest",
    interp_dim: str = "slant_range_time",
    **interp_kwargs: T.Any,
) -> xr.DataArray:
    """Interpolate the input image with optional multilook."""

    if multilook:
        image = image.rolling(
            azimuth_time=multilook[0], slant_range_time=multilook[1]
        ).mean()

    interp_kwargs[interp_dim] = range
    geocoded = image.interp(
        azimuth_time=azimuth_time, method=interp_method, **interp_kwargs
    )

    return geocoded.drop_vars(["azimuth_time", "ground_range", "pixel", "line"])


def terrain_correction_block(
    dem_raster: xr.DataArray,
    position_ecef: xr.DataArray,
    correct_radiometry: T.Optional[str],
    measurement_attrs: T.Dict[str, T.Any],
    slant_range_time0: float,
    azimuth_time0: float,
    coordinate_conversion: xr.Dataset,
    grouping_area_factor: T.Tuple[float, float],
) -> xr.Dataset:
    try:
        dem_ecef = scene.convert_to_dem_ecef(dem_raster)
    except Exception:
        dem_ecef = scene.convert_to_dem_ecef(dem_raster)
    dem_ecef = dem_ecef.drop_vars(dem_ecef.rio.grid_mapping)
    acquisition = simulate_acquisition(dem_ecef, position_ecef)
    acquisition = acquisition.drop_vars(["dem_direction", "axis"])

    if measurement_attrs["product_type"] == "GRD":
        ground_range = xarray_sentinel.slant_range_time_to_ground_range(
            acquisition.azimuth_time,
            acquisition.slant_range_time,
            coordinate_conversion,
        )
        acquisition["ground_range"] = ground_range.drop_vars("azimuth_time")
        acquisition = acquisition.drop_vars("slant_range_time")

    if correct_radiometry is not None:
        logger.info("correct radiometry")
        grid_parameters = radiometry.azimuth_slant_range_grid(
            measurement_attrs,
            slant_range_time0,
            azimuth_time0,
            measurement_attrs["product_type"] == "GRD",
            grouping_area_factor,
        )

        if correct_radiometry == "gamma_bilinear":
            gamma_weights = radiometry.gamma_weights_bilinear
        elif correct_radiometry == "gamma_nearest":
            gamma_weights = radiometry.gamma_weights_nearest

        weights = gamma_weights(
            dem_ecef,
            acquisition,
            **grid_parameters,
        )
        acquisition["weights"] = weights

    return acquisition


def terrain_correction(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    orbit_group: T.Optional[str] = None,
    calibration_group: T.Optional[str] = None,
    output_urlpath: str = "GTC.tif",
    correct_radiometry: T.Optional[str] = None,
    interp_method: str = "nearest",
    multilook: T.Optional[T.Tuple[int, int]] = None,
    grouping_area_factor: T.Tuple[float, float] = (3.0, 13.0),
    open_dem_raster_kwargs: T.Dict[str, T.Any] = {"chunks": 2048},
    chunks: T.Optional[T.Union[int, T.Dict[str, int]]] = None,
    **kwargs: T.Any,
) -> xr.DataArray:
    """Apply the terrain-correction to sentinel-1 SLC and GRD products.

    :param product_urlpath: input product path or url
    :param measurement_group: group of the measurement to be used, for example: "IW/VV"
    :param dem_urlpath: dem path or url
    :param orbit_group: overrides the orbit group name
    :param calibration_group: overridez the calibration group name
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
    :param kwargs: additional keyword arguments passed on to ``xarray.open_dataset`` to open the `product_urlpath`
    """
    allowed_correct_radiometry = [None, "gamma_bilinear", "gamma_nearest"]
    if correct_radiometry not in allowed_correct_radiometry:
        raise ValueError(
            f"{correct_radiometry=} not supported, "
            f"allowed values are: {allowed_correct_radiometry}"
        )

    orbit_group = orbit_group or f"{measurement_group}/orbit"
    calibration_group = calibration_group or f"{measurement_group}/calibration"

    logger.info(f"open data {product_urlpath!r}")

    measurement_ds, kwargs = open_dataset_autodetect(
        product_urlpath,
        group=measurement_group,
        chunks=chunks,
        **kwargs,  #  type: ignore
    )
    measurement = measurement_ds["measurement"]

    logger.info(f"open data {dem_urlpath!r} {rioxarray.__version__}")  # type: ignore

    dem_raster = scene.open_dem_raster(dem_urlpath, **open_dem_raster_kwargs)

    orbit_ecef = xr.open_dataset(product_urlpath, engine="sentinel-1", group=orbit_group, **kwargs)  # type: ignore
    position_ecef = orbit_ecef.position
    calibration = xr.open_dataset(product_urlpath, engine="sentinel-1", group=calibration_group, **kwargs)  # type: ignore
    beta_nought_lut = calibration.betaNought

    # clean dask templates
    template_raster = xr.zeros_like(dem_raster.drop_vars(dem_raster.rio.grid_mapping))
    template_acquisition = xr.Dataset(
        data_vars={
            "azimuth_time": template_raster.astype("datetime64[ns]"),
        }
    )

    if measurement.attrs["product_type"] == "GRD":
        coordinate_conversion = xr.open_dataset(
            product_urlpath,
            engine="sentinel-1",  # type: ignore
            group=f"{measurement_group}/coordinate_conversion",
            **kwargs,
        )
        slant_range_time0 = coordinate_conversion.slant_range_time.values[0]
        template_acquisition["ground_range"] = template_raster
    else:
        slant_range_time0 = measurement.slant_range_time.values[0]
        template_acquisition["slant_range_time"] = template_raster

    if correct_radiometry is not None:
        template_acquisition["weights"] = template_raster

    logger.info("calibrate radiometry")

    acquisition = xr.map_blocks(
        terrain_correction_block,
        dem_raster,
        kwargs={
            "position_ecef": position_ecef,
            "correct_radiometry": correct_radiometry,
            "measurement_attrs": measurement.attrs,
            "slant_range_time0": slant_range_time0,
            "azimuth_time0": measurement.azimuth_time.values[0],
            "coordinate_conversion": coordinate_conversion,
            "grouping_area_factor": grouping_area_factor,
        },
        template=template_acquisition,
    )

    beta_nought = xarray_sentinel.calibrate_intensity(measurement, beta_nought_lut)
    if measurement.attrs["product_type"] == "GRD":
        interp_arg = acquisition.ground_range
        interp_dim = "ground_range"
    elif measurement.attrs["product_type"] == "SLC":
        interp_arg = acquisition.slant_range_time
        interp_dim = "slant_range_time"
        if measurement.attrs["mode"] == "IW":
            beta_nought = xarray_sentinel.mosaic_slc_iw(beta_nought)
    else:
        raise ValueError(
            f"unsupported product_type {measurement.attrs['product_type']}"
        )

    geocoded = xr.map_blocks(
        interpolate_measurement,
        acquisition.azimuth_time,
        args=(interp_arg,),
        kwargs=dict(
            image=beta_nought,
            multilook=multilook,
            interp_method=interp_method,
            interp_dim=interp_dim,
        ),
        template=template_raster,
    )

    if correct_radiometry is not None:
        geocoded = geocoded / acquisition.weights

    logger.info("save output")

    geocoded.attrs.update(beta_nought.attrs)
    geocoded.x.attrs.update(dem_raster.x.attrs)
    geocoded.y.attrs.update(dem_raster.y.attrs)
    geocoded.rio.set_crs(dem_raster.rio.crs)
    geocoded.rio.to_raster(
        output_urlpath,
        dtype=np.float32,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="ZSTD",
        num_threads="ALL_CPUS",
    )

    return geocoded
