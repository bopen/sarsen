from typing import Any, Optional, Tuple, Literal

import attrs
import numpy as np
import pandas as pd
import xarray as xr

S_TO_NS = 10**9


def polyder(coefficients: xr.DataArray, poly_type: str) -> xr.DataArray:
    # TODO: raise if "degree" coord is not decreasing
    derivative_coefficients = coefficients.isel(degree=slice(1, None)).copy()
    if poly_type == 'polynomial':
        for degree in coefficients.coords["degree"].values[:-1]:
            derivative_coefficients.loc[{"degree": degree - 1}] = (
                coefficients.loc[{"degree": degree}] * degree
            )
    elif poly_type == 'hermite':
        v = [np.polynomial.hermite.Hermite(coefficients.isel(
            axis=i).values).deriv(m=1).coef for i in range(3)]
        derivative_coefficients.data = np.vstack(v).T
    return derivative_coefficients


@attrs.define
class OrbitPolyfitIterpolator:
    coefficients: xr.DataArray
    poly_type: str
    epoch: np.datetime64
    interval: Tuple[np.datetime64, np.datetime64]

    @classmethod
    def from_position(
        cls,
        position: xr.DataArray,
        dim: str = "azimuth_time",
        deg: int = 5,
        epoch: Optional[np.datetime64] = None,
        poly_type: Literal['polynomial', 'hermite'] = 'polynomial',
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
        if poly_type == 'polynomial':
            polyfit_results = data.polyfit(dim=dim, deg=deg)
        elif poly_type == 'hermite':
            v = [np.polynomial.hermite.hermfit(((time - epoch)/10**9).astype('float64'),
                                               data.values[i], deg=deg) for i in range(3)]
            polyfit_results = xr.Dataset(
                {'axis': [0, 1, 2], 'degree': np.arange(deg, -1, -1)})
            polyfit_results = polyfit_results.assign({'polyfit_coefficients':
                                                      (['degree', 'axis'], np.vstack(v).T)})
        # TODO: raise if the fit is not good enough
        return cls(polyfit_results.polyfit_coefficients, poly_type, epoch, interval)

    def azimuth_time_range(self, freq_s: float = 0.02) -> xr.DataArray:
        azimuth_time_values = pd.date_range(
            start=self.interval[0],
            end=self.interval[-1],
            freq=pd.Timedelta(freq_s, "s"),  # type: ignore
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
        if self.poly_type == 'polynomial':
            position = xr.polyval(time - self.epoch, self.coefficients)
            position = position.assign_coords(
                {time.name: time}).rename('position')
        elif self.poly_type == 'hermite':
            v = [np.polynomial.hermite.hermval(((time - self.epoch)/10**9).astype('float64'),
                                               self.coefficients.isel(axis=i)) for i in range(3)]
            position = xr.DataArray(data=np.vstack(v), dims=['axis', time.name],
                                    coords={'axis': [0, 1, 2], time.name: time}, name='position')
        return position

    def velocity(
        self, time: Optional[xr.DataArray] = None, **kwargs: Any
    ) -> xr.DataArray:
        if time is None:
            time = self.azimuth_time_range(**kwargs)
        assert time.dtype.name in ("datetime64[ns]", "timedelta64[ns]")

        velocity_coefficients = polyder(
            self.coefficients, self.poly_type) * S_TO_NS

        velocity: xr.DataArray
        if self.poly_type == 'polynomial':
            velocity = xr.polyval(time - self.epoch, velocity_coefficients)
            velocity = velocity.assign_coords(
                {time.name: time}).rename('velocity')
        elif self.poly_type == 'hermite':
            v = [np.polynomial.hermite.hermval(((time - self.epoch)/10**9).astype('float64'),
                                               velocity_coefficients.isel(axis=i)) for i in range(3)]
            velocity = xr.DataArray(data=np.vstack(v), dims=['axis', time.name],
                                    coords={'axis': [0, 1, 2], time.name: time}, name='velocity')
        return velocity
