import typing as T

import numpy as np
import xarray as xr
from rasterio import warp

# Earth-Centered, Earth-Fixed Coordinate Reference System used to express satellite orbit
# https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system
# https://spatialreference.org/ref/epsg/wgs-84-2/
ECEF_CRS = "EPSG:4978"


def open_dem_raster(dem_urlpath: str, **kwargs: T.Any) -> xr.DataArray:
    dem_raster = xr.open_dataarray(dem_urlpath, engine="rasterio", **kwargs)  # type: ignore
    if dem_raster.y.diff("y").values[0] < 0:
        dem_raster = dem_raster.isel(y=slice(None, None, -1))
    return dem_raster.squeeze(drop=True)  # type: ignore


def convert_to_dem_3d(dem_raster: xr.DataArray, dim: str = "axis") -> xr.DataArray:
    _, dem_raster_x = xr.broadcast(dem_raster, dem_raster.x)  # type: ignore
    dem_3d = xr.concat(
        [dem_raster_x, dem_raster.y, dem_raster], dim=dim, coords="minimal"
    )
    if dem_3d.chunks is not None:
        dem_3d = dem_3d.chunk({dim: None})

    dem_3d = dem_3d.assign_coords({dim: [0, 1, 2]})  # type: ignore
    return dem_3d.rename("dem_3d")


def transform_dem_3d(
    dem_3d: xr.DataArray,
    target_crs: str = ECEF_CRS,
    source_crs: T.Optional[str] = None,
    dim: str = "axis",
) -> xr.DataArray:
    if source_crs is None:
        source_crs = dem_3d.rio.crs
    x, y, z = warp.transform(
        source_crs,
        target_crs,
        dem_3d.sel({dim: 0}).values.flat,
        dem_3d.sel({dim: 1}).values.flat,
        dem_3d.sel({dim: 2}).values.flat,
    )
    dem_3d_crs: xr.DataArray = xr.zeros_like(dem_3d)
    shape = dem_3d_crs.loc[{dim: 0}].shape
    dem_3d_crs.loc[{dim: 0}] = np.reshape(x, shape)
    dem_3d_crs.loc[{dim: 1}] = np.reshape(y, shape)
    dem_3d_crs.loc[{dim: 2}] = np.reshape(z, shape)
    return dem_3d_crs


def convert_to_dem_ecef(dem_raster: xr.DataArray) -> xr.DataArray:
    dem_3d = convert_to_dem_3d(dem_raster)
    return transform_dem_3d(dem_3d, target_crs=ECEF_CRS)


def compute_diff_normal(
    dem_ecef: xr.DataArray,
    spatial_dims: T.Tuple[str, str] = ("x", "y"),
    axis_dim: str = "axis",
) -> xr.DataArray:
    x, y = spatial_dims
    x_diff = dem_ecef.shift({x: -1}) - dem_ecef.shift({x: 1})
    y_diff = dem_ecef.shift({y: -1}) - dem_ecef.shift({y: 1})
    up = xr.cross(x_diff, y_diff, dim=axis_dim)
    return up / xr.dot(up, up, dims=axis_dim) ** 0.5  # type: ignore
