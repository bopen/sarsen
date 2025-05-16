import functools
from typing import Any

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


def seconds_to_datetime64(time: xr.DataArray, epoch: np.datetime64) -> xr.DataArray:
    return time * np.timedelta64(S_TO_NS, "ns") + epoch


def datetime64_to_seconds(time: xr.DataArray, epoch: np.datetime64) -> xr.DataArray:
    return (time - epoch) / np.timedelta64(S_TO_NS, "ns")


@attrs.define
class OrbitPolyfitInterpolator:
    coefficients: xr.DataArray
    epoch: np.datetime64
    interval: tuple[np.datetime64, np.datetime64]

    @classmethod
    def from_position(
        cls,
        position: xr.DataArray,
        dim: str = "azimuth_time",
        deg: int = 5,
        epoch: np.datetime64 | None = None,
        interval: tuple[np.datetime64, np.datetime64] | None = None,
    ) -> "OrbitPolyfitInterpolator":
        time = position.coords[dim]

        if epoch is None:
            # NOTE: summing two datetime64 is not defined and we cannot use:
            #   `(time[0] + time[-1]) / 2` directly
            epoch = time.values[0] + (time.values[-1] - time.values[0]) / 2

        if interval is None:
            interval = (time.values[0], time.values[-1])

        seconds = datetime64_to_seconds(time, epoch)
        data = position.assign_coords({dim: seconds})
        polyfit_results = data.polyfit(dim=dim, deg=deg)
        # TODO: raise if the fit is not good enough

        return cls(polyfit_results.polyfit_coefficients, epoch, interval)

    def datetime64_to_seconds(self, time: xr.DataArray) -> xr.DataArray:
        return datetime64_to_seconds(time, self.epoch)

    def seconds_to_datetime64(self, time: xr.DataArray) -> xr.DataArray:
        return seconds_to_datetime64(time, self.epoch)

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

    def position_from_seconds(self, seconds: xr.DataArray) -> xr.DataArray:
        position = xr.polyval(seconds, self.coefficients)
        return position.rename("position")

    def position(self, time: xr.DataArray | None = None, **kwargs: Any) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        position = self.position_from_seconds(self.datetime64_to_seconds(time))
        return position.assign_coords({time.name: time})

    @functools.cached_property
    def velocity_coefficients(self) -> xr.DataArray:
        return polyder(self.coefficients)

    def velocity_from_seconds(self, seconds: xr.DataArray) -> xr.DataArray:
        velocity = xr.polyval(seconds, self.velocity_coefficients)
        return velocity.rename("velocity")

    def velocity(self, time: xr.DataArray | None = None, **kwargs: Any) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        velocity = self.velocity_from_seconds(self.datetime64_to_seconds(time))
        return velocity.assign_coords({time.name: time})

    @functools.cached_property
    def acceleration_coefficients(self) -> xr.DataArray:
        return polyder(self.velocity_coefficients)

    def acceleration_from_seconds(self, seconds: xr.DataArray) -> xr.DataArray:
        velocity = xr.polyval(seconds, self.acceleration_coefficients)
        return velocity.rename("acceleration")

    def acceleration(
        self, time: xr.DataArray | None = None, **kwargs: Any
    ) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        acceleration = self.acceleration_from_seconds(self.datetime64_to_seconds(time))
        return acceleration.assign_coords({time.name: time})


# keep wrong spelling used elsewhere
OrbitPolyfitIterpolator = OrbitPolyfitInterpolator
