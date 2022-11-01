from __future__ import annotations

import functools
from typing import Any, Dict, Optional, Tuple, Union

import attrs
import xarray as xr
import xarray_sentinel

from . import datamodel


def open_dataset_autodetect(
    product_urlpath: str,
    group: Optional[str] = None,
    chunks: Optional[Union[int, Dict[str, int]]] = None,
    **kwargs: Any,
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    kwargs.setdefault("engine", "sentinel-1")
    try:
        ds = xr.open_dataset(product_urlpath, group=group, chunks=chunks, **kwargs)
    except FileNotFoundError:
        # re-try with Planetary Computer option
        kwargs[
            "override_product_files"
        ] = "{dirname}/{prefix}{swath}-{polarization}{ext}"
        ds = xr.open_dataset(product_urlpath, group=group, chunks=chunks, **kwargs)
    return ds, kwargs


def calibrate_measurement(
    measurement_ds: xr.Dataset, beta_nought_lut: xr.DataArray
) -> xr.DataArray:
    measurement = measurement_ds.measurement
    if measurement.attrs["product_type"] == "SLC" and measurement.attrs["mode"] == "IW":
        measurement = xarray_sentinel.mosaic_slc_iw(measurement)

    beta_nought = xarray_sentinel.calibrate_intensity(measurement, beta_nought_lut)
    beta_nought = beta_nought.drop_vars(["pixel", "line"])

    return beta_nought


@attrs.define(slots=False)
class Sentinel1SarProduct(datamodel.SarProduct):
    product_urlpath: str
    measurement_group: str
    measurement_chunks: int = 2048
    kwargs: Dict[str, Any] = {}

    @functools.cached_property
    def measurement(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath,
            group=self.measurement_group,
            chunks=self.measurement_chunks,
            **self.kwargs,
        )
        return ds

    @functools.cached_property
    def orbit(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath, group=f"{self.measurement_group}/orbit", **self.kwargs
        )
        return ds

    @functools.cached_property
    def calibration(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath,
            group=f"{self.measurement_group}/calibration",
            **self.kwargs,
        )
        return ds

    @functools.cached_property
    def coordinate_conversion(self) -> Optional[xr.Dataset]:
        try:
            ds, self.kwargs = open_dataset_autodetect(
                self.product_urlpath,
                group=f"{self.measurement_group}/coordinate_conversion",
                **self.kwargs,
            )
        except TypeError:
            ds = None
        return ds

    def beta_nought(self) -> xr.DataArray:
        return calibrate_measurement(self.measurement, self.calibration.betaNought)


def product_info(
    product_urlpath: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get information about the Sentinel-1 product."""
    root_ds = xr.open_dataset(
        product_urlpath, engine="sentinel-1", check_files_exist=True, **kwargs
    )

    measurement_groups = [g for g in root_ds.attrs["subgroups"] if g.count("/") == 1]

    gcp_group = measurement_groups[0] + "/gcp"

    gcp, kwargs = open_dataset_autodetect(product_urlpath, group=gcp_group, **kwargs)

    bbox = [
        gcp.attrs["geospatial_lon_min"],
        gcp.attrs["geospatial_lat_min"],
        gcp.attrs["geospatial_lon_max"],
        gcp.attrs["geospatial_lat_max"],
    ]

    product_attrs = [
        "product_type",
        "mode",
        "swaths",
        "transmitter_receiver_polarisations",
    ]
    product_info = {attr_name: root_ds.attrs[attr_name] for attr_name in product_attrs}
    product_info.update(
        {
            "measurement_groups": measurement_groups,
            "geospatial_bounds": gcp.attrs["geospatial_bounds"],
            "geospatial_bbox": bbox,
        }
    )

    return product_info
