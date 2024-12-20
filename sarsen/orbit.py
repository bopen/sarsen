from typing import Any, Optional, Tuple

import attrs
import numpy as np
import pandas as pd
import xarray as xr

S_TO_NS = 10**9


def polyder(coefficients: xr.DataArray) -> xr.DataArray:
    # TODO: raise if "degree" coord is not decreasing
    derivative_coefficients = coefficients.isel(degree=slice(1, None)).copy()
    for degree in coefficients.coords["degree"].values[:-1]:
        derivative_coefficients.loc[{"degree": degree - 1}] = (
            coefficients.loc[{"degree": degree}] * degree
        )
    return derivative_coefficients


@attrs.define
class OrbitPolyfitIterpolator:
    coefficients: xr.DataArray
    epoch: np.datetime64
    interval: Tuple[np.datetime64, np.datetime64]

    @classmethod
    def from_position(
        cls,
        position: xr.DataArray,
        dim: str = "azimuth_time",
        deg: int = 5,
        epoch: Optional[np.datetime64] = None,
        interval: Optional[Tuple[np.datetime64, np.datetime64]] = None,
    ) -> "OrbitPolyfitIterpolator":
        time = position.coords[dim]
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        if epoch is None:
            # NOTE: summing two datetime64 is not defined and we cannot use:
            #   `(time[0] + time[-1]) / 2` directly
            epoch = time.values[0] + (time.values[-1] - time.values[0]) / 2

        if interval is None:
            interval = (time.values[0], time.values[-1])

        data = position.assign_coords({dim: time - epoch})
        polyfit_results = data.polyfit(dim=dim, deg=deg)
        # TODO: raise if the fit is not good enough

        return cls(polyfit_results.polyfit_coefficients, epoch, interval)

    def azimuth_time_range(self, freq_s: float = 0.02) -> xr.DataArray:
        azimuth_time_values = pd.date_range(
            start=self.interval[0],
            end=self.interval[-1],
            freq=pd.Timedelta(freq_s, "s"),
        ).values
        return xr.DataArray(
            azimuth_time_values,
            coords={"azimuth_time": azimuth_time_values},
            name="azimuth_time",
        )

    def position(
        self, time: Optional[xr.DataArray] = None, **kwargs: Any
    ) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        position: xr.DataArray
        position = xr.polyval(time - self.epoch, self.coefficients)
        position = position.assign_coords({time.name: time})
        return position.rename("position")

    def velocity(
        self, time: Optional[xr.DataArray] = None, **kwargs: Any
    ) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        velocity_coefficients = polyder(self.coefficients) * S_TO_NS

        velocity: xr.DataArray
        velocity = xr.polyval(time - self.epoch, velocity_coefficients)
        velocity = velocity.assign_coords({time.name: time})
        return velocity.rename("velocity")
