import typing as T

import attr
import numpy as np
import xarray as xr

M_OVER_NS_TO_M_OVER_S = 10 ** 9


def polyder(coefficients: xr.DataArray) -> xr.DataArray:
    # TODO: raise if "degree" coord is not decreasing
    derivative_coefficients = coefficients.isel(degree=slice(1, None)).copy()
    for degree in coefficients.coords["degree"].values[:-1]:
        derivative_coefficients.loc[{"degree": degree - 1}] = (
            coefficients.loc[{"degree": degree}] * degree
        )
    return derivative_coefficients


@attr.attrs(auto_attribs=True)
class OrbitPolyfitIterpolator:
    coefficients: xr.DataArray
    epoch: np.datetime64

    @classmethod
    def from_position(
        cls,
        position: xr.DataArray,
        dim: str = "azimuth_time",
        deg: int = 5,
        epoch: T.Optional[np.datetime64] = None,
    ) -> "OrbitPolyfitIterpolator":
        time = position.coords[dim]
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        if epoch is None:
            # NOTE: summing two datetime64 is not defined and we cannot use:
            #   `(time[0] + time[-1]) / 2` directly
            epoch = time.values[0] + (time.values[-1] - time.values[0]) / 2

        data = position.assign_coords({dim: time - epoch})  # type: ignore
        polyfit_results = data.polyfit(dim=dim, deg=deg)
        # TODO: raise if the fit is not good enough

        return cls(coefficients=polyfit_results.polyfit_coefficients, epoch=epoch)

    def position(self, time: xr.DataArray) -> xr.DataArray:
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")
        epoch_time = time.assign_coords({time.name: time - self.epoch})  # type: ignore

        position: xr.DataArray
        position = xr.polyval(epoch_time, self.coefficients)  # type: ignore
        position = position.assign_coords({time.name: time})  # type: ignore
        return position.rename("position")

    def velocity(self, time: xr.DataArray) -> xr.DataArray:
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")
        epoch_time = time.assign_coords({time.name: time - self.epoch})  # type: ignore

        velocity_coefficients = polyder(self.coefficients) * M_OVER_NS_TO_M_OVER_S

        velocity: xr.DataArray
        velocity = xr.polyval(epoch_time, velocity_coefficients)  # type: ignore
        velocity = velocity.assign_coords({time.name: time})  # type: ignore
        return velocity.rename("velocity")
