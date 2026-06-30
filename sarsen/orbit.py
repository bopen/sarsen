from typing import Any

import attrs
import numpy as np
import xarray as xr

from . import datamodel

S_TO_NS = 10**9


def polyder(coefficients: xr.DataArray) -> xr.DataArray:
    # TODO: raise if "degree" coord is not decreasing
    derivative_coefficients = coefficients.isel(degree=slice(1, None)).copy()
    for degree in coefficients.coords["degree"].values[:-1]:
        derivative_coefficients.loc[{"degree": degree - 1}] = (
            coefficients.loc[{"degree": degree}] * degree
        )
    return derivative_coefficients


def to_calendar_time(
    orbit_time: xr.DataArray, epoch: np.datetime64, name: str = "calendar_time"
) -> xr.DataArray:
    calendar_time = orbit_time * np.timedelta64(S_TO_NS, "ns") + epoch
    return calendar_time.rename(name)


def to_orbit_time(calendar_time: xr.DataArray, epoch: np.datetime64) -> xr.DataArray:
    orbit_time = (calendar_time - epoch) / np.timedelta64(S_TO_NS, "ns")
    return orbit_time.rename("orbit_time")


@attrs.define
class OrbitPolyfitInterpolator(datamodel.OrbitInterpolator):
    epoch: np.datetime64
    interval: tuple[np.datetime64, np.datetime64]
    position_coefficients: xr.DataArray
    velocity_coefficients: xr.DataArray
    acceleration_coefficients: xr.DataArray

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

        orbit_time = to_orbit_time(time, epoch)
        data = position.assign_coords({dim: orbit_time})
        polyfit_results = data.polyfit(dim=dim, deg=deg)
        # TODO: raise if the fit is not good enough

        position_coefficients = polyfit_results.polyfit_coefficients
        velocity_coefficients = polyder(position_coefficients)
        acceleration_coefficients = polyder(velocity_coefficients)
        self = cls(
            epoch,
            interval,
            position_coefficients,
            velocity_coefficients,
            acceleration_coefficients,
        )
        return self

    def to_calendar_time(self, orbit_time: xr.DataArray, **kwargs: Any) -> xr.DataArray:
        return to_calendar_time(orbit_time, self.epoch, **kwargs)

    def to_orbit_time(self, calendar_time: xr.DataArray, **kwargs: Any) -> xr.DataArray:
        return to_orbit_time(calendar_time, self.epoch, **kwargs)

    def position_from_orbit_time(self, orbit_time: xr.DataArray) -> xr.DataArray:
        position = xr.polyval(orbit_time, self.position_coefficients)
        return position.rename("position")

    def position(self, time: xr.DataArray, **kwargs: Any) -> xr.DataArray:
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        position = self.position_from_orbit_time(self.to_orbit_time(time))
        return position.assign_coords({time.name: time})

    def velocity_from_orbit_time(self, orbit_time: xr.DataArray) -> xr.DataArray:
        velocity = xr.polyval(orbit_time, self.velocity_coefficients)
        return velocity.rename("velocity")

    def velocity(self, time: xr.DataArray, **kwargs: Any) -> xr.DataArray:
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        velocity = self.velocity_from_orbit_time(self.to_orbit_time(time))
        return velocity.assign_coords({time.name: time})

    def acceleration_from_orbit_time(self, orbit_time: xr.DataArray) -> xr.DataArray:
        velocity = xr.polyval(orbit_time, self.acceleration_coefficients)
        return velocity.rename("acceleration")

    def acceleration(self, time: xr.DataArray, **kwargs: Any) -> xr.DataArray:
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        acceleration = self.acceleration_from_orbit_time(self.to_orbit_time(time))
        return acceleration.assign_coords({time.name: time})
