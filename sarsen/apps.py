import numpy as np
import xarray as xr

from . import geocoding, orbit, scene


def backward_geocode_sentinel1_slc_iw_burst(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str = "GRD.tif",
) -> None:
    swath_polarization_id, _, burst_index = measurement_group.rpartition("/")
    orbit_group = f"{swath_polarization_id}/orbit"

    print("open data")

    burst_sar = xr.open_dataset(product_urlpath, engine="sentinel-1", group=measurement_group)  # type: ignore
    orbit_ecef = xr.open_dataset(product_urlpath, engine="sentinel-1", group=orbit_group)  # type: ignore
    dem_raster = scene.open_dem_raster(dem_urlpath)

    print("process DEM")

    dem_ecef = scene.convert_to_dem_ecef(dem_raster)

    print("interpolate orbit")

    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(
        orbit_ecef.position
    )
    position_sar = orbit_interpolator.position(burst_sar.azimuth_time)
    velocity_sar = orbit_interpolator.velocity(burst_sar.azimuth_time)

    print("geocode")

    direction_sar = velocity_sar / np.linalg.norm(velocity_sar)
    dem_coords = geocoding.backward_geocode(dem_ecef, position_sar, direction_sar)

    print("interpolate")

    geocoded = abs(burst_sar).sel(
        azimuth_time=dem_coords.azimuth_time,
        slant_range_time=dem_coords.slant_range_time,
        method="nearest",
    )
    geocoded.rio.set_crs(dem_raster.rio.crs)

    geocoded.rio.to_raster(
        output_urlpath,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        num_threads="ALL_CPUS",
    )
