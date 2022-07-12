import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest import mock

import dask.array as da
import numpy as np
import rioxarray
import xarray as xr
import xarray_sentinel

from . import geocoding, orbit, radiometry, scene

logger = logging.getLogger(__name__)


def open_dataset_autodetect(
    product_urlpath: str,
    group: Optional[str] = None,
    chunks: Optional[Union[int, Dict[str, int]]] = None,
    **kwargs: Any,
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    try:
        ds = xr.open_dataset(
            product_urlpath, engine="sentinel-1", group=group, chunks=chunks, **kwargs
        )
    except FileNotFoundError:
        # re-try with Planetary Computer option
        kwargs[
            "override_product_files"
        ] = "{dirname}/{prefix}{swath}-{polarization}{ext}"
        ds = xr.open_dataset(
            product_urlpath, engine="sentinel-1", group=group, chunks=chunks, **kwargs
        )
    return ds, kwargs


def product_info(
    product_urlpath: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get information about the Sentinel-1 product."""
    root_ds = xr.open_dataset(product_urlpath, engine="sentinel-1", **kwargs)

    measurement_groups = [g for g in root_ds.attrs["subgroups"] if g.count("/") == 1]

    gcp_group = measurement_groups[0] + "/gcp"

    gcp, kwargs = open_dataset_autodetect(product_urlpath, group=gcp_group, **kwargs)

    bbox = [
        gcp.attrs["geospatial_lon_min"],
        gcp.attrs["geospatial_lat_min"],
        gcp.attrs["geospatial_lon_max"],
        gcp.attrs["geospatial_lat_max"],
    ]

    product_info = {
        "product_type": root_ds.attrs["product_type"],
        "mode": root_ds.attrs["mode"],
        "swaths": root_ds.attrs["swaths"],
        "transmitter_receiver_polarisations": root_ds.attrs[
            "transmitter_receiver_polarisations"
        ],
        "measurement_groups": measurement_groups,
        "bbox": bbox,
    }

    return product_info


def compute_chunks_1d(
    dim_size: int,
    chunks: int = 3000,
    bound: int = 30,
) -> Tuple[List[slice], List[slice], List[slice]]:
    ext_slices = []
    ext_slices_bound = []
    int_slices = []

    # -bound is needed to avoid to incorporate the last chunk, if smaller of bound in the previous chunk
    if dim_size > bound:
        number_of_chunks = int(np.ceil((dim_size - bound) / chunks))
    else:
        number_of_chunks = 1
    for n in np.arange(number_of_chunks):
        l_int = n * chunks
        if n * chunks - bound > 0:
            l_ext = n * chunks - bound
        else:
            l_ext = 0
        l_bound = l_int - l_ext

        if (n + 1) * chunks + bound < dim_size:
            r_ext = (n + 1) * chunks + bound
            r_int = (n + 1) * chunks
            r_bound = chunks + l_bound
        else:
            r_ext = dim_size
            r_int = dim_size
            r_bound = r_ext - l_ext

        ext_slices.append(slice(l_ext, r_ext))
        ext_slices_bound.append(slice(l_bound, r_bound))
        int_slices.append(slice(l_int, r_int))
    return ext_slices, ext_slices_bound, int_slices


def compute_product(
    slices: List[List[slice]], dims_name: List[str]
) -> List[Dict[str, slice]]:

    product: List[Dict[str, slice]] = []

    for slices_ in itertools.product(*slices):
        product.append({})
        for dim, sl in zip(dims_name, slices_):
            product[-1][dim] = sl
    return product


def compute_chunks(
    dims: Dict[str, int] = {},
    chunks: int = 2000,
    bound: int = 30,
) -> Tuple[List[Dict[str, slice]], List[Dict[str, slice]], List[Dict[str, slice]]]:
    ext_slices_ = []
    ext_slices_bound_ = []
    int_slices_ = []
    for dim_size in dims.values():
        ec, ecb, ic = compute_chunks_1d(dim_size, chunks=chunks, bound=bound)
        ext_slices_.append(ec)
        ext_slices_bound_.append(ecb)
        int_slices_.append(ic)

    ext_slices = compute_product(ext_slices_, list(dims))
    ext_slices_bound = compute_product(ext_slices_bound_, list(dims))
    int_slices = compute_product(int_slices_, list(dims))
    return ext_slices, ext_slices_bound, int_slices


def execute_on_overlapping_blocks(
    function: Callable[..., xr.DataArray],
    obj: Union[xr.Dataset, xr.DataArray],
    chunks: int = 3000,
    bound: int = 30,
    kwargs: Dict[Any, Any] = {},
    template: Optional[xr.DataArray] = None,
) -> xr.DataArray:

    dims = {}
    for d in obj.dims:
        dims[str(d)] = len(obj[d])

    if isinstance(obj, xr.Dataset):
        if template is None:
            raise ValueError(
                "template argument is mandatory if obj is type of xr.Dataset"
            )
    elif isinstance(obj, xr.DataArray):
        if template is None:
            template = obj

    ext_chunks, ext_chunks_bounds, int_chunks = compute_chunks(
        dims, chunks, bound
    )  # type ignore
    out = xr.DataArray(da.empty_like(template.data), dims=template.dims)  # type: ignore
    out.coords.update(obj.coords)
    for ext_chunk, ext_chunk_bounds, int_chunk in zip(
        ext_chunks, ext_chunks_bounds, int_chunks
    ):
        out_chunk = function(obj.isel(ext_chunk), **kwargs)
        out[int_chunk] = out_chunk.isel(ext_chunk_bounds)
    return out


def simulate_acquisition(
    dem_ecef: xr.DataArray,
    position_ecef: xr.DataArray,
    coordinate_conversion: Optional[xr.Dataset],
    correct_radiometry: Optional[str],
) -> xr.Dataset:
    """Compute the image coordinates of the DEM given the satellite orbit."""

    orbit_interpolator = orbit.OrbitPolyfitIterpolator.from_position(position_ecef)
    position_ecef = orbit_interpolator.position()
    velocity_ecef = orbit_interpolator.velocity()

    acquisition = geocoding.backward_geocode(dem_ecef, position_ecef, velocity_ecef)

    if coordinate_conversion is not None:
        ground_range = xarray_sentinel.slant_range_time_to_ground_range(
            acquisition.azimuth_time,
            acquisition.slant_range_time,
            coordinate_conversion,
        )
        acquisition["ground_range"] = ground_range.drop_vars("azimuth_time")
        if correct_radiometry is None:
            acquisition = acquisition.drop_vars("slant_range_time")
    if correct_radiometry is not None:
        gamma_area = radiometry.compute_gamma_area(dem_ecef, acquisition.dem_direction)
        acquisition["gamma_area"] = gamma_area

    acquisition = acquisition.drop_vars(["dem_direction", "axis"])

    return acquisition


def terrain_correction(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    orbit_group: Optional[str] = None,
    calibration_group: Optional[str] = None,
    output_urlpath: str = "GTC.tif",
    correct_radiometry: Optional[str] = None,
    interp_method: str = "nearest",
    grouping: Tuple[float, float] = (3.0, 3.0),
    oversampling: Tuple[float, float] = (1, 1),
    open_dem_raster_kwargs: Dict[str, Any] = {},
    chunks: Optional[int] = 1024,
    radiometry_chunks: int = 3000,
    radiometry_bound: int = 30,
    enable_dask_distributed: bool = False,
    client_kwargs: Dict[str, Any] = {"processes": False},
    **kwargs: Any,
) -> xr.DataArray:
    """Apply the terrain-correction to sentinel-1 SLC and GRD products.

    :param product_urlpath: input product path or url
    :param measurement_group: group of the measurement to be used, for example: "IW/VV"
    :param dem_urlpath: dem path or url
    :param orbit_group: overrides the orbit group name
    :param calibration_group: overridez the calibration group name
    :param output_urlpath: output path or url
    :param correct_radiometry: default `None`. If `correct_radiometry=None`the radiometric terrain
    correction is not applied. `correct_radiometry=gamma_bilinear` applies the gamma flattening classic
    algorithm using bilinear interpolation to compute the weights. `correct_radiometry=gamma_nearest`
    applies the gamma flattening using nearest neighbours instead of bilinear interpolation.
    'gamma_nearest' significantly reduces the processing time
    :param interp_method: interpolation method for product resampling.
    The interpolation methods are the methods supported by ``xarray.DataArray.interp``
    :param multilook: multilook factor. If `None` the multilook is not applied
    :param grouping: is a tuple of floats greater than 1. The default is `(1, 1)`.
    The `grouping_area_factor`  can be increased (i) to speed up the processing or
    (ii) when the input DEM resolution is low.
    The Gamma Flattening usually works properly if the pixel size of the input DEM is much smaller
    than the pixel size of the input Sentinel-1 product. Otherwise, the output may have radiometric distortions.
    This problem can be avoided by increasing the `grouping_area_factor`.
    Be aware that `grouping_area_factor` too high may degrade the final result
    :param open_dem_raster_kwargs: additional keyword arguments passed on to ``xarray.open_dataset``
    to open the `dem_urlpath`
    :param kwargs: additional keyword arguments passed on to ``xarray.open_dataset`` to open the `product_urlpath`
    """
    # rioxarray must be imported explicitly or accesses to `.rio` may fail in dask
    assert rioxarray.__version__  # type: ignore

    allowed_correct_radiometry = [None, "gamma_bilinear", "gamma_nearest"]
    if correct_radiometry not in allowed_correct_radiometry:
        raise ValueError(
            f"{correct_radiometry=} not supported, "
            f"allowed values are: {allowed_correct_radiometry}"
        )

    orbit_group = orbit_group or f"{measurement_group}/orbit"
    calibration_group = calibration_group or f"{measurement_group}/calibration"

    output_chunks = chunks if chunks is not None else 512

    to_raster_kwargs = {}
    if enable_dask_distributed:
        from dask.distributed import Client, Lock

        client = Client(**client_kwargs)
        to_raster_kwargs["lock"] = Lock("rio", client=client)
        to_raster_kwargs["compute"] = False
        print(f"Dask distributed dashboard at: {client.dashboard_link}")

    logger.info(f"open data {product_urlpath!r}")

    measurement_ds, kwargs = open_dataset_autodetect(
        product_urlpath,
        group=measurement_group,
        chunks=1024,
        **kwargs,
    )
    measurement = measurement_ds["measurement"]

    dem_raster = scene.open_dem_raster(
        dem_urlpath, chunks=chunks, **open_dem_raster_kwargs
    )

    orbit_ecef = xr.open_dataset(
        product_urlpath, engine="sentinel-1", group=orbit_group, **kwargs
    )
    position_ecef = orbit_ecef.position
    calibration = xr.open_dataset(
        product_urlpath, engine="sentinel-1", group=calibration_group, **kwargs
    )
    beta_nought_lut = calibration.betaNought

    if measurement.attrs["product_type"] == "GRD":
        coordinate_conversion = xr.open_dataset(
            product_urlpath,
            engine="sentinel-1",
            group=f"{measurement_group}/coordinate_conversion",
            **kwargs,
        )
    else:
        coordinate_conversion = None

    template_raster = dem_raster.drop_vars(dem_raster.rio.grid_mapping)

    logger.info("pre-process DEM")

    dem_ecef = xr.map_blocks(
        scene.convert_to_dem_ecef, dem_raster, kwargs={"source_crs": dem_raster.rio.crs}
    )
    dem_ecef = dem_ecef.drop_vars(dem_ecef.rio.grid_mapping)

    acquisition_template = xr.Dataset(
        data_vars={
            "slant_range_time": template_raster,
            "azimuth_time": (template_raster * 0).astype("datetime64[ns]"),
        }
    )
    if coordinate_conversion is not None:
        acquisition_template["ground_range"] = template_raster
        if correct_radiometry is None:
            acquisition_template = acquisition_template.drop_vars("slant_range_time")
    if correct_radiometry is not None:
        acquisition_template["gamma_area"] = template_raster

    logger.info("simulate acquisition")

    acquisition = xr.map_blocks(
        simulate_acquisition,
        dem_ecef,
        kwargs={
            "position_ecef": position_ecef,
            "coordinate_conversion": coordinate_conversion,
            "correct_radiometry": correct_radiometry,
        },
        template=acquisition_template,
    )

    if measurement.attrs["product_type"] == "GRD":
        assert coordinate_conversion is not None
        slant_range_time0 = coordinate_conversion.slant_range_time.values[0]
        interp_kwargs = {"ground_range": acquisition.ground_range}
    elif measurement.attrs["product_type"] == "SLC":
        slant_range_time0 = measurement.slant_range_time.values[0]
        interp_kwargs = {"slant_range_time": acquisition.slant_range_time}
        if measurement.attrs["mode"] == "IW":
            measurement = xarray_sentinel.mosaic_slc_iw(measurement)
    else:
        raise ValueError(
            f"unsupported product_type {measurement.attrs['product_type']}"
        )

    if correct_radiometry is not None:
        logger.info("simulate radiometry")
        grid_parameters = radiometry.azimuth_slant_range_grid(
            measurement.attrs,
            slant_range_time0,
            measurement.azimuth_time.values[0],
            grouping,
        )

        if correct_radiometry == "gamma_bilinear":
            gamma_weights = radiometry.gamma_weights_bilinear
        elif correct_radiometry == "gamma_nearest":
            gamma_weights = radiometry.gamma_weights_nearest

        grid_parameters["oversampling"] = oversampling

        acquisition = acquisition.persist()

        with mock.patch("xarray.core.missing._localize", lambda o, i: (o, i)):
            weights = execute_on_overlapping_blocks(
                obj=acquisition,
                function=gamma_weights,
                chunks=radiometry_chunks,
                bound=radiometry_bound,
                kwargs=grid_parameters,
                template=acquisition["gamma_area"],
            )

    logger.info("calibrate image")

    beta_nought = xarray_sentinel.calibrate_intensity(measurement, beta_nought_lut)
    beta_nought = beta_nought.drop_vars(["pixel", "line"])

    logger.info("terrain-correct image")

    # HACK: we monkey-patch away an optimisation in xr.DataArray.interp that actually makes
    #   the interpolation much slower when indices are dask arrays.
    with mock.patch("xarray.core.missing._localize", lambda o, i: (o, i)):
        geocoded = beta_nought.interp(
            method=interp_method,  # type: ignore
            azimuth_time=acquisition.azimuth_time,
            **interp_kwargs,
        )

    if correct_radiometry is not None:
        geocoded = geocoded / weights

    logger.info("save output")

    geocoded.attrs.update(beta_nought.attrs)
    geocoded.x.attrs.update(dem_raster.x.attrs)
    geocoded.y.attrs.update(dem_raster.y.attrs)
    geocoded.rio.set_crs(dem_raster.rio.crs)
    maybe_delayed = geocoded.rio.to_raster(
        output_urlpath,
        dtype=np.float32,
        tiled=True,
        blockxsize=output_chunks,
        blockysize=output_chunks,
        compress="ZSTD",
        num_threads="ALL_CPUS",
        **to_raster_kwargs,
    )

    if enable_dask_distributed:
        maybe_delayed.compute()
        client.close()

    return geocoded
