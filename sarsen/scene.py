import typing as T

import numpy as np
import xarray as xr
from rasterio import warp


def make_dem_3d(dem_da: xr.DataArray, dim: str = "axis") -> xr.DataArray:
    _, dem_da_x = xr.broadcast(dem_da, dem_da.x)  # type: ignore
    dem_3d = xr.concat([dem_da_x, dem_da.y, dem_da], dim=dim, coords="minimal")
    dem_3d = dem_3d.assign_coords({dim: ["x", "y", "z"]})  # type: ignore
    return dem_3d.rename("dem_3d")


def transform_dem_3d(
    dem_crs: xr.DataArray,
    target_crs: str = "EPSG:4978",
    source_crs: T.Optional[str] = None,
    dim: str = "axis",
) -> xr.DataArray:
    if source_crs is None:
        source_crs = dem_crs.rio.crs
    x, y, z = warp.transform(
        source_crs,
        target_crs,
        dem_crs.sel({dim: "x"}).values.flat,
        dem_crs.sel({dim: "y"}).values.flat,
        dem_crs.sel({dim: "z"}).values.flat,
    )
    dem_ecef_crs: xr.DataArray = xr.zeros_like(dem_crs)
    shape = dem_ecef_crs.loc[{dim: "x"}].shape
    dem_ecef_crs.loc[{dim: "x"}] = np.reshape(x, shape)
    dem_ecef_crs.loc[{dim: "y"}] = np.reshape(y, shape)
    dem_ecef_crs.loc[{dim: "z"}] = np.reshape(z, shape)
    return dem_ecef_crs
