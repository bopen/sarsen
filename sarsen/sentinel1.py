import functools
from typing import Any

import attrs
import numpy as np
import xarray as xr
import xarray_sentinel

import sarsen

try:
    import dask  # noqa: F401

    DEFAULT_MEASUREMENT_CHUNKS: int | None = 2048
except ModuleNotFoundError:
    DEFAULT_MEASUREMENT_CHUNKS = None

SPEED_OF_LIGHT = 299_792_458.0  # m / s


def open_dataset_autodetect(
    product_urlpath: str,
    group: str | None = None,
    chunks: int | dict[str, int] | None = None,
    check_files_exist: bool = False,
    **kwargs: Any,
) -> tuple[xr.Dataset, dict[str, Any]]:
    kwargs.setdefault("engine", "sentinel-1")
    try:
        ds = xr.open_dataset(
            product_urlpath,
            group=group,
            chunks=chunks,
            check_files_exist=check_files_exist,
            **kwargs,
        )
    except FileNotFoundError:
        # re-try with Planetary Computer option
        kwargs["override_product_files"] = (
            "{dirname}/{prefix}{swath}-{polarization}{ext}"
        )
        ds = xr.open_dataset(product_urlpath, group=group, chunks=chunks, **kwargs)
    return ds, kwargs


def azimuth_slant_range_grid(
    attrs: dict[str, Any],
    grouping_area_factor: tuple[float, float] = (3.0, 3.0),
) -> dict[str, Any]:
    if attrs["product_type"] == "SLC":
        slant_range_spacing_m = (
            attrs["range_pixel_spacing"]
            * np.sin(attrs["incidence_angle_mid_swath"])
            * grouping_area_factor[1]
        )
    else:
        slant_range_spacing_m = attrs["range_pixel_spacing"] * grouping_area_factor[1]

    slant_range_time_interval_s = (
        slant_range_spacing_m * 2 / SPEED_OF_LIGHT  # ignore type
    )

    grid_parameters: dict[str, Any] = {
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
class Sentinel1SarProduct(sarsen.GroundRangeSarProduct, sarsen.SlantRangeSarProduct):
    product_urlpath: str
    measurement_group: str | None = None
    measurement_chunks: int | None = DEFAULT_MEASUREMENT_CHUNKS
    kwargs: dict[str, Any] = {}

    def all_measurement_groups(self) -> list[str]:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath, check_files_exist=True, **self.kwargs
        )

        measurement_groups = [g for g in ds.attrs["subgroups"] if g.count("/") == 1]
        return measurement_groups

    @property
    def measurement(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath,
            group=self.measurement_group,
            chunks=self.measurement_chunks,
            **self.kwargs,
        )
        if ds.attrs["product_type"] == "SLC" and ds.attrs["mode"] == "IW":
            ds = xarray_sentinel.mosaic_slc_iw(ds)
        return ds

    @functools.cached_property
    def orbit(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath, group=f"{self.measurement_group}/orbit", **self.kwargs
        )
        return ds.compute()

    @functools.cached_property
    def gcp(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath, group=f"{self.measurement_group}/gcp", **self.kwargs
        )
        return ds.compute()

    @functools.cached_property
    def calibration(self) -> xr.Dataset:
        ds, self.kwargs = open_dataset_autodetect(
            self.product_urlpath,
            group=f"{self.measurement_group}/calibration",
            **self.kwargs,
        )
        return ds.compute()

    @functools.cached_property
    def coordinate_conversion(self) -> xr.Dataset | None:
        ds = None
        if self.product_type == "GRD":
            ds, self.kwargs = open_dataset_autodetect(
                self.product_urlpath,
                group=f"{self.measurement_group}/coordinate_conversion",
                **self.kwargs,
            )
            ds = ds.compute()
        return ds

    @functools.cached_property
    def azimuth_fm_rate(self) -> xr.Dataset | None:
        ds = None
        if self.product_type == "SLC":
            ds, self.kwargs = open_dataset_autodetect(
                self.product_urlpath,
                group=f"{self.measurement_group}/azimuth_fm_rate",
                **self.kwargs,
            )
            ds = ds.compute()
        return ds

    @functools.cached_property
    def dc_estimate(self) -> xr.Dataset | None:
        ds = None
        if self.product_type == "SLC":
            ds, self.kwargs = open_dataset_autodetect(
                self.product_urlpath,
                group=f"{self.measurement_group}/dc_estimate",
                **self.kwargs,
            )
            ds = ds.compute()
        return ds

    # make class hashable to allow caching methods calls

    def __hash__(self) -> int:
        id = (
            self.product_urlpath,
            self.measurement_group,
            self.measurement_chunks,
        ) + tuple(repr(self.kwargs))
        return hash(id)

    # SarProduct interaface

    @functools.cached_property
    def product_type(self) -> Any:
        prod_type = self.measurement.attrs["product_type"]
        assert isinstance(prod_type, str)
        return prod_type

    @functools.cache
    def beta_nought(self, persist: bool = False) -> xr.DataArray:
        measurement = self.measurement.data_vars["measurement"]
        beta_nought = xarray_sentinel.calibrate_intensity(
            measurement, self.calibration.betaNought
        )
        if persist:
            beta_nought = beta_nought.persist()
        return beta_nought.drop_vars(["pixel", "line"])

    def state_vectors(self) -> xr.DataArray:
        return self.orbit.data_vars["position"]

    def slant_range_time_to_ground_range(
        self, azimuth_time: xr.DataArray, slant_range_time: xr.DataArray
    ) -> xr.DataArray:
        assert self.coordinate_conversion is not None
        # the dask graph explodes without the map_blocks due to the interpolations
        ground_range = xr.map_blocks(
            xarray_sentinel.slant_range_time_to_ground_range,
            azimuth_time,
            args=(slant_range_time,),
            kwargs={"coordinate_conversion": self.coordinate_conversion},
        )
        return ground_range

    def grid_parameters(
        self,
        grouping_area_factor: tuple[float, float] = (3.0, 3.0),
    ) -> dict[str, Any]:
        return azimuth_slant_range_grid(self.measurement.attrs, grouping_area_factor)

    def complex_amplitude(self) -> xr.DataArray:
        measurement = self.measurement.data_vars["measurement"]
        beta_nought = xarray_sentinel.calibrate_amplitude(
            measurement, self.calibration.betaNought
        )
        beta_nought = beta_nought.drop_vars(["pixel", "line"])
        return beta_nought

    def interp_sar(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        if self.product_type == "GRD":
            return sarsen.GroundRangeSarProduct.interp_sar(self, *args, **kwargs)
        else:
            return sarsen.SlantRangeSarProduct.interp_sar(self, *args, **kwargs)

    @functools.cache
    def product_info(self, **kwargs: Any) -> dict[str, Any]:
        """Get information about the Sentinel-1 product."""
        measurement_groups = self.all_measurement_groups()

        self.measurement_group = self.measurement_group or measurement_groups[0]
        gcp = self.gcp

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
            "relative_orbit_number",
            "orbit_number",
            "mission_data_take_id",
        ]
        product_info = {}
        for attr_name in product_attrs:
            try:
                product_info[attr_name] = gcp.attrs[attr_name]
            except KeyError:
                pass
        product_info.update(
            {
                "measurement_groups": measurement_groups,
                "geospatial_bounds": gcp.attrs["geospatial_bounds"],
                "geospatial_bbox": bbox,
            }
        )

        return product_info
