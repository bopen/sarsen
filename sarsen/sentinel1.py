from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import attrs
import numpy as np
import xarray as xr
import xarray_sentinel

from . import datamodel, geocoding

try:
    import dask  # noqa: F401

    DEFAULT_MEASUREMENT_CHUNKS: Optional[int] = 2048
except ModuleNotFoundError:
    DEFAULT_MEASUREMENT_CHUNKS = None


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
    measurement: xr.DataArray, beta_nought_lut: xr.DataArray
) -> xr.DataArray:
    if measurement.attrs["product_type"] == "SLC" and measurement.attrs["mode"] == "IW":
        measurement = xarray_sentinel.mosaic_slc_iw(measurement)

    beta_nought = xarray_sentinel.calibrate_intensity(measurement, beta_nought_lut)
    beta_nought = beta_nought.drop_vars(["pixel", "line"])

    return beta_nought


def azimuth_slant_range_grid(
    attrs: Dict[str, Any],
    grouping_area_factor: Tuple[float, float] = (3.0, 3.0),
) -> Dict[str, Any]:

    if attrs["product_type"] == "SLC":
        slant_range_spacing_m = (
            attrs["range_pixel_spacing"]
            * np.sin(attrs["incidence_angle_mid_swath"])
            * grouping_area_factor[1]
        )
    else:
        slant_range_spacing_m = attrs["range_pixel_spacing"] * grouping_area_factor[1]

    slant_range_time_interval_s = (
        slant_range_spacing_m * 2 / geocoding.SPEED_OF_LIGHT  # ignore type
    )

    grid_parameters: Dict[str, Any] = {
        "slant_range_time0": attrs["image_slant_range_time"],
        "slant_range_time_interval_s": slant_range_time_interval_s,
        "slant_range_spacing_m": slant_range_spacing_m,
        "azimuth_time0": np.datetime64(attrs["product_first_line_utc_time"]),
        "azimuth_time_interval_s": attrs["azimuth_time_interval"]
        * grouping_area_factor[0],
        "azimuth_spacing_m": attrs["azimuth_pixel_spacing"] * grouping_area_factor[0],
    }
    return grid_parameters


@attrs.define(slots=False)
class Sentinel1SarProduct(datamodel.SarProduct):
    product_urlpath: str
    measurement_group: str
    measurement_chunks: Optional[int] = DEFAULT_MEASUREMENT_CHUNKS
    kwargs: Dict[str, Any] = {}

    @property
    def measurement(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath,
            group=self.measurement_group,
            chunks=self.measurement_chunks,
            **self.kwargs,
        )
        return ds

    @property
    def orbit(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath, group=f"{self.measurement_group}/orbit", **self.kwargs
        )
        return ds

    @property
    def calibration(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath,
            group=f"{self.measurement_group}/calibration",
            **self.kwargs,
        )
        return ds

    @property
    def coordinate_conversion(self) -> Optional[xr.Dataset]:
        ds = None
        if self.product_type == "GRD":
            ds, self.kwargs = open_dataset_autodetect(
                self.product_urlpath,
                group=f"{self.measurement_group}/coordinate_conversion",
                **self.kwargs,
            )
        return ds

    @property
    def azimuth_fm_rate(self) -> Optional[xr.Dataset]:
        ds = None
        if self.product_type == "SLC":
            ds, self.kwargs = open_dataset_autodetect(
                self.product_urlpath,
                group=f"{self.measurement_group}/azimuth_fm_rate",
                **self.kwargs,
            )
        return ds

    @property
    def dc_estimate(self) -> Optional[xr.Dataset]:
        ds = None
        if self.product_type == "SLC":
            ds, self.kwargs = open_dataset_autodetect(
                self.product_urlpath,
                group=f"{self.measurement_group}/dc_estimate",
                **self.kwargs,
            )
        return ds

    # SarProduct interaface

    @property
    def product_type(self) -> Any:
        prod_type = self.measurement.attrs["product_type"]
        assert isinstance(prod_type, str)
        return prod_type

    def beta_nought(self) -> xr.DataArray:
        measurement = self.measurement.data_vars["measurement"]
        return calibrate_measurement(measurement, self.calibration.betaNought)

    def state_vectors(self) -> xr.DataArray:
        return self.orbit.data_vars["position"]

    def slant_range_time_to_ground_range(
        self, azimuth_time: xr.DataArray, slant_range_time: xr.DataArray
    ) -> Optional[xr.DataArray]:
        ds = None
        coordinate_conversion = self.coordinate_conversion
        if coordinate_conversion is not None:
            ds = xarray_sentinel.slant_range_time_to_ground_range(
                azimuth_time, slant_range_time, coordinate_conversion
            )
        return ds

    def grid_parameters(
        self,
        grouping_area_factor: Tuple[float, float] = (3.0, 3.0),
    ) -> Dict[str, Any]:
        return azimuth_slant_range_grid(self.measurement.attrs, grouping_area_factor)


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
