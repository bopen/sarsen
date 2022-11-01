from __future__ import annotations

import abc

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
