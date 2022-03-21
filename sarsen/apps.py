import logging
import typing as T

import numpy as np
import xarray as xr
import xarray_sentinel

from . import geocoding, orbit, radiometry, scene

logger = logging.getLogger(__name__)


def simulate_acquisition(
    position_ecef: xr.DataArray,
    dem_ecef: xr.DataArray,
) -> xr.Dataset:

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
    if multilook:
        image = image.rolling(
            azimuth_time=multilook[0], slant_range_time=multilook[1]
        ).mean()

    geocoded = image.interp(method=interp_method, **interp_kwargs)

    return geocoded


def check_dem_resolution(
    dem_ecef: xr.DataArray,
    slant_range_spacing_m: float,
    azimuth_spacing_m: float,
) -> None:
    dem_area = abs(dem_ecef.x[1] - dem_ecef.x[0]) * abs(dem_ecef.y[1] - dem_ecef.y[0])
    grouping_area = slant_range_spacing_m * azimuth_spacing_m
    if grouping_area / dem_area < 2 ** 2:
        logger.warning(
            "DEM resolution is too low, "
            "consider to over-sample the input DEM or to a use an higher ´grouping_area_factor´"
        )


def backward_geocode_sentinel1(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    orbit_group: T.Optional[str] = None,
    calibration_group: T.Optional[str] = None,
    output_urlpath: str = "GRD.tif",
    correct_radiometry: bool = False,
    interp_method: str = "nearest",
    multilook: T.Optional[T.Tuple[int, int]] = None,
    grouping_area_factor: T.Tuple[float, float] = (3.0, 13.0),
    open_dem_raster_kwargs: T.Dict[str, T.Any] = {},
    **kwargs: T.Any,
) -> xr.DataArray:
    if correct_radiometry and "chunks" in kwargs:
        raise ValueError("chunks are not supported if ´correct_radiometry´ is True")

    orbit_group = orbit_group or f"{measurement_group}/orbit"
    calibration_group = calibration_group or f"{measurement_group}/calibration"

    logger.info(f"open data {product_urlpath!r}")

    try:
        measurement_ds = xr.open_dataset(
            product_urlpath, engine="sentinel-1", group=measurement_group, **kwargs  # type: ignore
        )
    except FileNotFoundError:
        # re-try with Planetary Computer option
        kwargs = {
            "override_product_files": "{dirname}/{prefix}{swath}-{polarization}{ext}"
        }
        measurement_ds = xr.open_dataset(
            product_urlpath, engine="sentinel-1", group=measurement_group, **kwargs  # type: ignore
        )

    measurement = measurement_ds.measurement

    dem_raster = scene.open_dem_raster(dem_urlpath, **open_dem_raster_kwargs)

    orbit_ecef = xr.open_dataset(product_urlpath, engine="sentinel-1", group=orbit_group, **kwargs)  # type: ignore
    position_ecef = orbit_ecef.position
    calibration = xr.open_dataset(product_urlpath, engine="sentinel-1", group=calibration_group, **kwargs)  # type: ignore
    beta_nought_lut = calibration.betaNought

    logger.info("pre-process DEM")

    dem_ecef = scene.convert_to_dem_ecef(dem_raster)

    logger.info("simulate acquisition")

    acquisition = simulate_acquisition(position_ecef, dem_ecef)

    logger.info("calibrate radiometry")

    beta_nought = xarray_sentinel.calibrate_intensity(measurement, beta_nought_lut)

    logger.info("interpolate image")
    coordinate_conversion = None
    if measurement_ds.attrs["sar:product_type"] == "GRD":
        coordinate_conversion = xr.open_dataset(
            product_urlpath,
            engine="sentinel-1",
            group=f"{measurement_group}/coordinate_conversion",
            **kwargs,
        )  # type: ignore
        ground_range = xarray_sentinel.slant_range_time_to_ground_range(
            acquisition.azimuth_time,
            acquisition.slant_range_time,
            coordinate_conversion,
        )
        interp_kwargs = {"ground_range": ground_range}
    elif measurement_ds.attrs["sar:product_type"] == "SLC":
        interp_kwargs = {"slant_range_time": acquisition.slant_range_time}
        if measurement_ds.attrs["sar:instrument_mode"] == "IW":
            beta_nought = xarray_sentinel.mosaic_slc_iw(beta_nought)
    else:
        raise ValueError(
            f"unsupported sar:product_type {measurement_ds.attrs['sar:product_type']}"
        )

    geocoded = interpolate_measurement(
        beta_nought,
        multilook=multilook,
        azimuth_time=acquisition.azimuth_time,
        interp_method=interp_method,
        **interp_kwargs,
    )

    if correct_radiometry:
        logger.info("correct radiometry")
        grid_parameters = radiometry.azimuth_slant_range_grid(
            measurement_ds, coordinate_conversion, grouping_area_factor
        )
        check_dem_resolution(
            dem_ecef,
            slant_range_spacing_m=grid_parameters["slant_range_spacing_m"],
            azimuth_spacing_m=grid_parameters["azimuth_spacing_m"],
        )
        weights = radiometry.gamma_weights(
            dem_ecef,
            acquisition,
            **grid_parameters,
        )
        geocoded = geocoded / weights

    logger.info("save output")

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
