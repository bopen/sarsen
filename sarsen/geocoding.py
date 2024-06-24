"""Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019.

See: https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""

import numpy as np
import numpy.typing as npt
import numpy.polynomial.polynomial as poly
import xarray as xr

MAXITER = 10

def backward_geocode(
    dem_ecef: xr.DataArray,
    pos: xr.DataArray,
    t0: xr.DataArray | np.datetime64 | None = None,
    dim: str = "axis",
    conv_th: float = 1.0e-6,
) -> xr.Dataset:

    assert pos.attrs.get("referenceTime", None) is not None, "orbit reference time not defined"
    assert pos.attrs.get("scaleTime", None) is not None, "orbit scaling time not defined"
    
    if (t0 is not None):
        if isinstance(t0, xr.DataArray):
            assert t0.size == dem_ecef.isel({dim: 0}).size, "number of guess points different from dem_ecef size"
            
            t = ((t0 - pos.referenceTime) * pos.scaleTime).astype(float)
        else:
            t = xr.full_like(dem_ecef.isel({dim: 0}).drop_vars(dim), (t0 - pos.referenceTime)*pos.scaleTime, dtype=float)
    else:
        t = xr.full_like(dem_ecef.isel({dim: 0}).drop_vars(dim), 0, dtype=float)
        
    # compute orbit polynomial derivatives
    vel = xr.DataArray(poly.polyder(pos, 1), dims=["degree", dim])
    acc = xr.DataArray(poly.polyder(vel, 1), dims=["degree", dim])
    
    vel = vel.assign_coords({"degree": np.arange(vel.degree.size)})
    acc = acc.assign_coords({"degree": np.arange(acc.degree.size)})
    
    for k in range(MAXITER):        
        # compute start point
        p = xr.polyval(t, pos)
        v = xr.polyval(t, vel)
        a = xr.polyval(t, acc)

        # compute range vector
        r = p - dem_ecef

        # update time
        F = (v*r).sum(dim=dim)
        F1 = (a*r).sum(dim=dim)+(v*v).sum(dim=dim)
        delta = (F / F1)

        maxcorr = np.abs(delta).max().values
        if (maxcorr < conv_th):
            break
        
        t = t - delta

    direction_ecef = (
        v / xr.dot(v, v, dims=dim) ** 0.5
    )

    acquisition = xr.Dataset(
        data_vars={
            "azimuth_time": (t / pos.scaleTime).astype("timedelta64[ns]")+pos.referenceTime,
            "dem_distance": -r,
            "satellite_direction": direction_ecef,
        }
    )
    
    return acquisition
