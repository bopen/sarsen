import typing as T

import numpy as np
import xarray as xr
import xarray_sentinel

from . import geocoding, orbit, scene


def backward_geocode_slc(
    measurement: xr.DataArray,
    position_ecef: xr.DataArray,
    dem_raster: xr.DataArray,
) -> xr.DataArray:

    print("pre-process DEM")

    dem_ecef = scene.convert_to_dem_ecef(dem_raster)

    print("interpolate orbit")

    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(position_ecef)
    position_ecef = orbit_interpolator.position()
    velocity_ecef = orbit_interpolator.velocity()
    direction_ecef = velocity_ecef / (velocity_ecef ** 2).sum("axis") ** 0.5

    print("geocode")

    dem_coords = geocoding.backward_geocode(dem_ecef, position_ecef, direction_ecef)

    print("interpolate")

    if "number_of_bursts" in measurement.attrs:
        geocoded = xr.full_like(dem_raster, np.nan)

        azimuth_time_min = dem_coords.azimuth_time.values.min()
        azimuth_time_max = dem_coords.azimuth_time.values.max()
        slant_range_time_min = dem_coords.slant_range_time.values.min()
        slant_range_time_max = dem_coords.slant_range_time.values.max()

        for burst_index in range(measurement.attrs["number_of_bursts"]):
            burst = xarray_sentinel.crop_burst_dataset(
                measurement, burst_index=burst_index
            )
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
            temp = abs(burst.isel(azimuth_time=slice(30, -30))).interp(
                azimuth_time=dem_coords.azimuth_time,
                slant_range_time=dem_coords.slant_range_time,
                method="nearest",
            )
            geocoded = xr.where(np.isfinite(temp), temp, geocoded)  # type: ignore

    else:
        geocoded = abs(measurement).interp(
            azimuth_time=dem_coords.azimuth_time,
            slant_range_time=dem_coords.slant_range_time,
            method="nearest",
        )

    return geocoded


def backward_geocode_sentinel1_slc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    orbit_group: T.Optional[str] = None,
    output_urlpath: str = "GRD.tif",
) -> None:
    orbit_group = orbit_group or f"{measurement_group}/orbit"

    print("open data")

    measurement = xr.open_dataarray(product_urlpath, engine="sentinel-1", group=measurement_group)  # type: ignore
    orbit_ecef = xr.open_dataset(product_urlpath, engine="sentinel-1", group=orbit_group)  # type: ignore
    position_ecef = orbit_ecef.position
    dem_raster = scene.open_dem_raster(dem_urlpath)

    print("process data")

    geocoded = backward_geocode_slc(measurement, position_ecef, dem_raster)

    print("save data")

    geocoded.rio.set_crs(dem_raster.rio.crs)
    geocoded.rio.to_raster(
        output_urlpath,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="ZSTD",
        num_threads="ALL_CPUS",
    )
