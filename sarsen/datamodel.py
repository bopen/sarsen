import abc
from typing import Any, Optional

import xarray as xr


class SarProduct(abc.ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    @abc.abstractmethod
    def product_type(self) -> str:
        ...

    @abc.abstractmethod
    def state_vectors(self) -> xr.DataArray:
        ...

    @abc.abstractmethod
    def beta_nought(self) -> xr.DataArray:
        ...

    @abc.abstractmethod
    def beta_nought_interp(
        self,
        azimuth_time: xr.DataArray,
        slant_range_time: xr.DataArray,
        method: xr.core.types.InterpOptions = "nearest",
    ) -> xr.DataArray:
        ...


class GroundRangeSarProduct(SarProduct):
    def beta_nought_interp(
        self,
        azimuth_time: xr.DataArray,
        slant_range_time: xr.DataArray,
        method: xr.core.types.InterpOptions = "nearest",
    ) -> xr.DataArray:
        beta_nought = self.beta_nought()
        ground_range = self.slant_range_time_to_ground_range(
            azimuth_time, slant_range_time
        )
        interpolated = beta_nought.interp(
            azimuth_time=azimuth_time, ground_range=ground_range, method=method
        )
        return interpolated.assign_attrs(beta_nought.attrs)

    @abc.abstractmethod
    def slant_range_time_to_ground_range(
        self, azimuth_time: xr.DataArray, slant_range_time: xr.DataArray
    ) -> Optional[xr.DataArray]:
        ...


class SlantRangeSarProduct(SarProduct):
    @abc.abstractmethod
    def complex_amplitude(self) -> xr.DataArray:
        ...

    def beta_nought(self) -> xr.DataArray:
        amplitude = self.complex_amplitude()
        assert amplitude.attrs["units"] == "m m-1"
        beta_nought = abs(amplitude) ** 2
        return beta_nought.assign_attrs(long_name="beta nought", units="m2 m-2")

    def beta_nought_interp(
        self,
        azimuth_time: xr.DataArray,
        slant_range_time: xr.DataArray,
        method: xr.core.types.InterpOptions = "nearest",
    ) -> xr.DataArray:
        beta_nought = self.beta_nought()
        interpolated = beta_nought.interp(
            azimuth_time=azimuth_time, slant_range_time=slant_range_time, method=method
        )
        return interpolated.assign_attrs(beta_nought.attrs)
