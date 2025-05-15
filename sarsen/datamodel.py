import abc
from typing import Any

import xarray as xr


class SarProduct(abc.ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    @abc.abstractmethod
    def product_type(self) -> str: ...

    @abc.abstractmethod
    def state_vectors(self) -> xr.DataArray: ...

    @abc.abstractmethod
    def beta_nought(self) -> xr.DataArray: ...

    @abc.abstractmethod
    def interp_sar(
        self,
        data: xr.DataArray,
        azimuth_time: xr.DataArray,
        slant_range_time: xr.DataArray | None = None,
        method: xr.core.types.InterpOptions = "nearest",
        ground_range: xr.DataArray | None = None,
    ) -> xr.DataArray: ...

    # FIXME: design a better interface
    @abc.abstractmethod
    def grid_parameters(
        self,
        grouping_area_factor: tuple[float, float] = (3.0, 3.0),
    ) -> dict[str, Any]: ...


class GroundRangeSarProduct(SarProduct):
    @abc.abstractmethod
    def slant_range_time_to_ground_range(
        self, azimuth_time: xr.DataArray, slant_range_time: xr.DataArray
    ) -> xr.DataArray: ...

    def interp_sar(
        self,
        data: xr.DataArray,
        azimuth_time: xr.DataArray,
        slant_range_time: xr.DataArray | None = None,
        method: xr.core.types.InterpOptions = "nearest",
        ground_range: xr.DataArray | None = None,
    ) -> xr.DataArray:
        if ground_range is None:
            assert slant_range_time is not None
            ground_range = self.slant_range_time_to_ground_range(
                azimuth_time, slant_range_time
            )
        interpolated = data.interp(
            azimuth_time=azimuth_time, ground_range=ground_range, method=method
        )
        return interpolated.assign_attrs(data.attrs)


class SlantRangeSarProduct(SarProduct):
    @abc.abstractmethod
    def complex_amplitude(self) -> xr.DataArray: ...

    def beta_nought(self) -> xr.DataArray:
        amplitude = self.complex_amplitude()
        assert amplitude.attrs["units"] == "m m-1"
        beta_nought = abs(amplitude) ** 2
        return beta_nought.assign_attrs(long_name="beta nought", units="m2 m-2")

    def interp_sar(
        self,
        data: xr.DataArray,
        azimuth_time: xr.DataArray,
        slant_range_time: xr.DataArray | None = None,
        method: xr.core.types.InterpOptions = "nearest",
        ground_range: xr.DataArray | None = None,
    ) -> xr.DataArray:
        assert ground_range is None
        interpolated = data.interp(
            azimuth_time=azimuth_time, slant_range_time=slant_range_time, method=method
        )
        return interpolated.assign_attrs(data.attrs)
