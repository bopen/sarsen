import abc
from typing import Any, Dict, Optional, Tuple

import xarray as xr


class SarProduct(abc.ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

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

    # FIXME: design a better interface
    def grid_parameters(
        self,
        grouping_area_factor: Tuple[float, float] = (3.0, 3.0),
    ) -> Dict[str, Any]:
        ...
