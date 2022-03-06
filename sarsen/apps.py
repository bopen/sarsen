import typing as T

import numpy as np
import xarray as xr
import xarray_sentinel

from . import geocoding, orbit, scene


def simulate_acquisition(
    position_ecef: xr.DataArray,
    dem_raster: xr.DataArray,
) -> xr.Dataset:

    print("pre-process DEM")

    dem_ecef = scene.convert_to_dem_ecef(dem_raster)

    print("interpolate orbit")

    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(position_ecef)
    position_ecef = orbit_interpolator.position()
    velocity_ecef = orbit_interpolator.velocity()

    print("geocode")

    acquisition = geocoding.backward_geocode(dem_ecef, position_ecef, velocity_ecef)

    return acquisition


def interpolate_measurement(
    image: xr.DataArray,
    acquisition: xr.Dataset,
    dem_raster: xr.DataArray,
    multilook: T.Optional[T.Tuple[int, int]] = (2, 8),
    interp_method: str = "nearest",
    **interp_kwargs: T.Any,
) -> xr.DataArray:
    if "number_of_bursts" in image.attrs:
        geocoded = xr.full_like(dem_raster, np.nan)

        azimuth_time_min = acquisition.azimuth_time.values.min()
        azimuth_time_max = acquisition.azimuth_time.values.max()
        slant_range_time_min = acquisition.slant_range_time.values.min()
        slant_range_time_max = acquisition.slant_range_time.values.max()

        for burst_index in range(image.attrs["number_of_bursts"]):
            burst = xarray_sentinel.crop_burst_dataset(image, burst_index=burst_index)
            if multilook:
                burst = burst.rolling(
                    azimuth_time=multilook[0],
                    slant_range_time=multilook[1],
                    center=True,
                ).mean()
            if (
                burst.azimuth_time[-1] < azimuth_time_min
                or burst.azimuth_time[0] > azimuth_time_max
            ):
                continue
            if (
                burst.slant_range_time[-1] < slant_range_time_min
                or burst.slant_range_time[0] > slant_range_time_max
            ):
                continue
            # the `isel` is very crude way to remove the black bands in azimuth
            temp = burst.isel(azimuth_time=slice(30, -30)).interp(
                azimuth_time=acquisition.azimuth_time,
                slant_range_time=acquisition.slant_range_time,
                method=interp_method,
            )
            geocoded = xr.where(np.isfinite(temp), temp, geocoded)  # type: ignore
    else:
        geocoded = image.interp(
            method=interp_method,
            **interp_kwargs,
        )

    return geocoded


def backward_geocode_sentinel1(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    orbit_group: T.Optional[str] = None,
    calibration_group: T.Optional[str] = None,
    output_urlpath: str = "GRD.tif",
    correct_radiometry: bool = False,
    interp_method: str = "nearest",
    multilook: T.Optional[T.Tuple[int, int]] = (2, 8),
) -> None:
    orbit_group = orbit_group or f"{measurement_group}/orbit"
    calibration_group = calibration_group or f"{measurement_group}/calibration"

    print("open data")

    measurement_ds = xr.open_dataset(product_urlpath, engine="sentinel-1", group=measurement_group)  # type: ignore
    product_type = measurement_ds.attrs["sar:product_type"]
    measurement = measurement_ds.measurement

    dem_raster = scene.open_dem_raster(dem_urlpath)

    orbit_ecef = xr.open_dataset(product_urlpath, engine="sentinel-1", group=orbit_group)  # type: ignore
    position_ecef = orbit_ecef.position
    calibration = xr.open_dataset(product_urlpath, engine="sentinel-1", group=calibration_group)  # type: ignore
    beta_nought_lut = calibration.beta_naught

    print("simulate acquisition")

    acquisition = simulate_acquisition(position_ecef, dem_raster)

    print("calibrate radiometry")

    beta_nought = xarray_sentinel.calibrate_intensity(measurement, beta_nought_lut)

    print("interpolate image")

    if product_type == "GRD":
        coordinate_conversion = xr.open_dataset(
            product_urlpath,
            engine="sentinel-1",
            group=f"{measurement_group}/coordinate_conversion",
        )  # type: ignore
        ground_range = xarray_sentinel.slant_range_time_to_ground_range(
            acquisition.azimuth_time,
            acquisition.aslant_range_time,
            coordinate_conversion,
        )
        interp_kwargs = {"ground_range": ground_range}
    else:
        interp_kwargs = {"slant_range_time": acquisition.slant_range_time}

    geocoded = interpolate_measurement(
        beta_nought,
        acquisition,
        dem_raster,
        multilook=multilook,
        azimuth_time=acquisition.azimuth_time,
        method=interp_method,
        **interp_kwargs,
    )

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
