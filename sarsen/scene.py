import typing as T

import numpy as np
import xarray as xr
from rasterio import warp


def make_dem_3d(dem_raster: xr.DataArray, dim: str = "axis") -> xr.DataArray:
    _, dem_raster_x = xr.broadcast(dem_raster, dem_raster.x)  # type: ignore
    dem_3d = xr.concat(
        [dem_raster_x, dem_raster.y, dem_raster], dim=dim, coords="minimal"
    )
    dem_3d = dem_3d.assign_coords({dim: [0, 1, 2]})  # type: ignore
    return dem_3d.rename("dem_3d")


def transform_dem_3d(
    dem_3d: xr.DataArray,
    target_crs: str = "EPSG:4978",
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
