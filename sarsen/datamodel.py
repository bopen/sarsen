from __future__ import annotations

import abc
from typing import Optional

import xarray as xr


class SarProduct(abc.ABC):
    @property
    @abc.abstractmethod
    def product_type(self) -> str:
        ...

    @abc.abstractmethod
    def beta_nought(self) -> xr.DataArray:
        ...

    @abc.abstractmethod
    def state_vectors(self) -> xr.DataArray:
        ...

    def slant_range_time_to_ground_range(
        self, azimuth_time: xr.DataArray, slant_range_time: xr.DataArray
    ) -> Optional[xr.DataArray]:
        return None
