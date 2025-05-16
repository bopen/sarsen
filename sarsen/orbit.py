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


def orbit_time_to_azimuth_time(
    orbit_time: xr.DataArray, epoch: np.datetime64
) -> xr.DataArray:
    azimuth_time = orbit_time * np.timedelta64(S_TO_NS, "ns") + epoch
    return azimuth_time.rename("azimuth_time")


def azimuth_time_to_orbit_time(
    azimuth_time: xr.DataArray, epoch: np.datetime64
) -> xr.DataArray:
    orbit_time = (azimuth_time - epoch) / np.timedelta64(S_TO_NS, "ns")
    return orbit_time.rename("orbit_time")


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

        orbit_time = azimuth_time_to_orbit_time(time, epoch)
        data = position.assign_coords({dim: orbit_time})
        polyfit_results = data.polyfit(dim=dim, deg=deg)
        # TODO: raise if the fit is not good enough

        return cls(polyfit_results.polyfit_coefficients, epoch, interval)

    def orbit_time_to_azimuth_time(self, azimuth_time: xr.DataArray) -> xr.DataArray:
        return orbit_time_to_azimuth_time(azimuth_time, self.epoch)

    def azimuth_time_to_orbit_time(self, orbit_time: xr.DataArray) -> xr.DataArray:
        return azimuth_time_to_orbit_time(orbit_time, self.epoch)

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

    def position_from_orbit_time(self, orbit_time: xr.DataArray) -> xr.DataArray:
        position = xr.polyval(orbit_time, self.coefficients)
        return position.rename("position")

    def position(self, time: xr.DataArray | None = None, **kwargs: Any) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        position = self.position_from_orbit_time(self.azimuth_time_to_orbit_time(time))
        return position.assign_coords({time.name: time})

    @functools.cached_property
    def velocity_coefficients(self) -> xr.DataArray:
        return polyder(self.coefficients)

    def velocity_from_orbit_time(self, orbit_time: xr.DataArray) -> xr.DataArray:
        velocity = xr.polyval(orbit_time, self.velocity_coefficients)
        return velocity.rename("velocity")

    def velocity(self, time: xr.DataArray | None = None, **kwargs: Any) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        velocity = self.velocity_from_orbit_time(self.azimuth_time_to_orbit_time(time))
        return velocity.assign_coords({time.name: time})

    @functools.cached_property
    def acceleration_coefficients(self) -> xr.DataArray:
        return polyder(self.velocity_coefficients)

    def acceleration_from_orbit_time(self, orbit_time: xr.DataArray) -> xr.DataArray:
        velocity = xr.polyval(orbit_time, self.acceleration_coefficients)
        return velocity.rename("acceleration")

    def acceleration(
        self, time: xr.DataArray | None = None, **kwargs: Any
    ) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        acceleration = self.acceleration_from_orbit_time(
            self.azimuth_time_to_orbit_time(time)
        )
        return acceleration.assign_coords({time.name: time})


# keep wrong spelling used elsewhere
OrbitPolyfitIterpolator = OrbitPolyfitInterpolator
