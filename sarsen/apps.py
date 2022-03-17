import typing as T

import numpy as np
import xarray as xr
import xarray_sentinel

from . import geocoding, orbit, scene


def mosaic_slc_iw(image: xr.DataArray, crop: int = 90) -> xr.DataArray:
    bursts = []
    for i in range(image.attrs["number_of_bursts"]):
        burst = xarray_sentinel.crop_burst_dataset(image, burst_index=i)
        bursts.append(burst.isel(azimuth_time=slice(crop, -crop)))
    return xr.concat(bursts, dim="azimuth_time")  # type: ignore


def simulate_acquisition(
    position_ecef: xr.DataArray,
    dem_ecef: xr.DataArray,
) -> xr.Dataset:

    print("interpolate orbit")

    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(position_ecef)
    position_ecef = orbit_interpolator.position()
    velocity_ecef = orbit_interpolator.velocity()

    print("geocode")

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


def backward_geocode_sentinel1(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    orbit_group: T.Optional[str] = None,
    calibration_group: T.Optional[str] = None,
    output_urlpath: str = "GRD.tif",
    correct_radiometry: T.Optional[str] = None,
    interp_method: str = "nearest",
    multilook: T.Optional[T.Tuple[int, int]] = None,
    **kwargs: T.Any,
) -> None:
    orbit_group = orbit_group or f"{measurement_group}/orbit"
    calibration_group = calibration_group or f"{measurement_group}/calibration"

    print("open data")

    measurement_ds = xr.open_dataset(product_urlpath, engine="sentinel-1", group=measurement_group, **kwargs)  # type: ignore
    measurement = measurement_ds.measurement

    dem_raster = scene.open_dem_raster(dem_urlpath)

    orbit_ecef = xr.open_dataset(product_urlpath, engine="sentinel-1", group=orbit_group, **kwargs)  # type: ignore
    position_ecef = orbit_ecef.position
    calibration = xr.open_dataset(product_urlpath, engine="sentinel-1", group=calibration_group, **kwargs)  # type: ignore
    beta_nought_lut = calibration.betaNought

    print("pre-process DEM")

    dem_ecef = scene.convert_to_dem_ecef(dem_raster)

    print("simulate acquisition")

    acquisition = simulate_acquisition(position_ecef, dem_ecef)

    print("calibrate radiometry")

    beta_nought = xarray_sentinel.calibrate_intensity(measurement, beta_nought_lut)

    print("interpolate image")

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
            beta_nought = mosaic_slc_iw(beta_nought)
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

    if correct_radiometry == "gamma":
        print("correct radiometry")
        if measurement_ds.attrs["sar:product_type"] == "GRD":
            slant_range_time0 = coordinate_conversion.slant_range_time.values[0]
        else:
            slant_range_time0 = measurement.slant_range_time.values[0]

        weights = geocoding.gamma_weights(
            dem_ecef.compute(),
            acquisition.compute(),
            slant_range_time0=slant_range_time0,
            azimuth_time0=measurement.azimuth_time.values[0],
            azimuth_time_interval=measurement.attrs["azimuth_time_interval"],
            slant_range_time_interval=measurement.attrs["slant_range_time_interval"],
            pixel_spacing_azimuth=measurement.attrs["sar:pixel_spacing_azimuth"],
            pixel_spacing_range=measurement.attrs["sar:pixel_spacing_range"],
        )
        geocoded = geocoded / weights

    elif correct_radiometry == "cosine":

        print("correct radiometry")
        if measurement_ds.attrs["sar:product_type"] == "GRD":
            slant_range_time0 = coordinate_conversion.slant_range_time.values[0]
        else:
            slant_range_time0 = measurement.slant_range_time.values[0]

        dem_normal = scene.compute_diff_normal(dem_ecef)

        cos_incidence_angle = xr.dot(dem_normal, -acquisition.dem_direction, dims="axis")  # type: ignore
        initial_weights = xr.where(cos_incidence_angle > 0, cos_incidence_angle, np.nan)

        weights_cosine, weights_count = geocoding.sum_weights(
            initial_weights.compute(),
            acquisition.compute(),
            slant_range_time0=slant_range_time0,
            azimuth_time0=measurement.azimuth_time.values[0],
            azimuth_time_interval=measurement.attrs["azimuth_time_interval"],
            slant_range_time_interval=measurement.attrs["slant_range_time_interval"],
        )
        # we don't really have an explanation for the `weights_count ** 0.5` term
        geocoded = geocoded / weights_cosine / weights_count ** 0.5
    elif correct_radiometry is not None:
        raise ValueError(f"unkwon radiometry corrcetion method: {correct_radiometry!r}")

    print("save output")

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
