import typing as T

import numpy as np
import numpy.typing as npt
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


def compute_dem_oriented_area(
    dem_ecef: xr.DataArray,
) -> xr.DataArray:
    x_corners: npt.ArrayLike = np.concatenate(
        [
            [dem_ecef.x[0] + (dem_ecef.x[0] - dem_ecef.x[1]) / 2],
            ((dem_ecef.x.shift(x=-1) + dem_ecef.x) / 2)[:-1].data,
            [dem_ecef.x[-1] + (dem_ecef.x[-1] - dem_ecef.x[-2]) / 2],
        ]
    )
    y_corners: npt.ArrayLike = np.concatenate(
        [
            [dem_ecef.y[0] + (dem_ecef.y[0] - dem_ecef.y[1]) / 2],
            ((dem_ecef.y.shift(y=-1) + dem_ecef.y) / 2)[:-1].data,
            [dem_ecef.y[-1] + (dem_ecef.y[-1] - dem_ecef.y[-2]) / 2],
        ]
    )
    dem_ecef_corners = dem_ecef.interp(
        {"x": x_corners, "y": y_corners},
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    dx = dem_ecef_corners.diff("x", 1)
    dy = dem_ecef_corners.diff("y", 1)

    dx1 = dx.isel(y=slice(1, None)).assign_coords(dem_ecef.coords)  # type: ignore
    dy1 = dy.isel(x=slice(1, None)).assign_coords(dem_ecef.coords)  # type: ignore
    dx2 = dx.isel(y=slice(None, -1)).assign_coords(dem_ecef.coords)  # type: ignore
    dy2 = dy.isel(x=slice(None, -1)).assign_coords(dem_ecef.coords)  # type: ignore

    cross_1 = xr.cross(dx1, dy1, dim="axis") / 2
    sign_1 = np.sign(
        xr.dot(cross_1, dem_ecef, dims="axis")  # type: ignore
    )  # ensure direction out of DEM

    cross_2 = xr.cross(dx2, dy2, dim="axis") / 2
    sign_2 = np.sign(
        xr.dot(cross_2, dem_ecef, dims="axis")  # type: ignore
    )  # ensure direction out of DEM
    dem_oriented_area: xr.DataArray = cross_1 * sign_1 + cross_2 * sign_2

    return dem_oriented_area
