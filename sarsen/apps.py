import logging
import typing as T

import numpy as np
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
            product_urlpath, engine="sentinel-1", group=group, chunks=chunks, **kwargs
        )
    except FileNotFoundError:
        # re-try with Planetary Computer option
        kwargs[
            "override_product_files"
        ] = "{dirname}/{prefix}{swath}-{polarization}{ext}"
        ds = xr.open_dataset(
            product_urlpath, engine="sentinel-1", group=group, chunks=chunks, **kwargs
        )
    return ds, kwargs


def product_info(
    product_urlpath: str,
    **kwargs: T.Any,
) -> T.Dict[str, T.Any]:
    """Get information about the Sentinel-1 product."""
    root_ds = xr.open_dataset(product_urlpath, engine="sentinel-1", **kwargs)

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
    image: xr.DataArray,
    multilook: T.Optional[T.Tuple[int, int]] = None,
    interp_method: str = "nearest",
    **interp_kwargs: T.Any,
) -> xr.DataArray:
    """Interpolate the input image with optional multilook."""

    if multilook:
        image = image.rolling(
            azimuth_time=multilook[0], slant_range_time=multilook[1]
        ).mean()

    geocoded = image.interp(method=interp_method, **interp_kwargs)  # type: ignore

    return geocoded


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
    grouping_area_factor: T.Tuple[float, float] = (3.0, 3.0),
    open_dem_raster_kwargs: T.Dict[str, T.Any] = {},
    chunks: T.Optional[int] = 1024,
    enable_dask_distributed: bool = False,
    client_kwargs: T.Dict[str, T.Any] = {"processes": False},
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

    output_chunks = chunks if chunks is not None else 512

    if enable_dask_distributed:
        from dask.distributed import Client

        client = Client(**client_kwargs)
        print(f"Dask distributed dashboard at: {client.dashboard_link}")

    logger.info(f"open data {product_urlpath!r}")

    measurement_ds, kwargs = open_dataset_autodetect(
        product_urlpath,
        group=measurement_group,
        chunks=chunks,
        **kwargs,  #  type: ignore
    )
    measurement = measurement_ds["measurement"]

    dem_raster = scene.open_dem_raster(dem_urlpath, **open_dem_raster_kwargs)

    orbit_ecef = xr.open_dataset(
        product_urlpath, engine="sentinel-1", group=orbit_group, **kwargs
    )
    position_ecef = orbit_ecef.position
    calibration = xr.open_dataset(
        product_urlpath, engine="sentinel-1", group=calibration_group, **kwargs
    )
    beta_nought_lut = calibration.betaNought

    if measurement.attrs["product_type"] == "GRD":
        coordinate_conversion = xr.open_dataset(
            product_urlpath,
            engine="sentinel-1",
            group=f"{measurement_group}/coordinate_conversion",
            **kwargs,
        )
    else:
        coordinate_conversion = None

    logger.info("pre-process DEM")

    dem_ecef = scene.convert_to_dem_ecef(dem_raster)

    logger.info("simulate acquisition")

    acquisition = simulate_acquisition(dem_ecef, position_ecef)

    logger.info("calibrate radiometry")

    beta_nought = xarray_sentinel.calibrate_intensity(measurement, beta_nought_lut)

    logger.info("interpolate image")
    if measurement.attrs["product_type"] == "GRD":
        assert coordinate_conversion is not None
        ground_range = xarray_sentinel.slant_range_time_to_ground_range(
            acquisition.azimuth_time,
            acquisition.slant_range_time,
            coordinate_conversion,
        )
        slant_range_time0 = coordinate_conversion.slant_range_time.values[0]
        interp_kwargs = {"ground_range": ground_range}
    elif measurement.attrs["product_type"] == "SLC":
        slant_range_time0 = measurement.slant_range_time.values[0]
        interp_kwargs = {"slant_range_time": acquisition.slant_range_time}
        if measurement.attrs["mode"] == "IW":
            beta_nought = xarray_sentinel.mosaic_slc_iw(beta_nought)
    else:
        raise ValueError(
            f"unsupported product_type {measurement.attrs['product_type']}"
        )

    geocoded = interpolate_measurement(
        beta_nought,
        multilook=multilook,
        azimuth_time=acquisition.azimuth_time,
        interp_method=interp_method,
        **interp_kwargs,
    )

    if correct_radiometry is not None:
        logger.info("correct radiometry")
        grid_parameters = radiometry.azimuth_slant_range_grid(
            measurement.attrs,
            slant_range_time0,
            measurement.azimuth_time.values[0],
            grouping_area_factor,
        )

        if correct_radiometry == "gamma_bilinear":
            gamma_weights = radiometry.gamma_weights_bilinear
        elif correct_radiometry == "gamma_nearest":
            gamma_weights = radiometry.gamma_weights_nearest

        weights = gamma_weights(
            dem_ecef.compute(),
            acquisition.compute(),
            **grid_parameters,
        )
        geocoded = geocoded / weights

    logger.info("save output")

    geocoded.attrs.update(beta_nought.attrs)
    geocoded.x.attrs.update(dem_raster.x.attrs)
    geocoded.y.attrs.update(dem_raster.y.attrs)
    geocoded.rio.set_crs(dem_raster.rio.crs)
    geocoded.rio.to_raster(
        output_urlpath,
        dtype=np.float32,
        tiled=True,
        blockxsize=output_chunks,
        blockysize=output_chunks,
        compress="ZSTD",
        num_threads="ALL_CPUS",
    )

    return geocoded
