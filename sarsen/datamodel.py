from __future__ import annotations

import abc

import xarray as xr


class SarProduct(abc.ABC):
    @abc.abstractmethod
    def beta_nought(self) -> xr.DataArray:
        ...
