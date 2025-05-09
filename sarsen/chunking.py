import itertools
import math
from typing import Any, Callable

import xarray as xr


def compute_chunks_1d(
    dim_size: int,
    chunks: int = 2048,
    bound: int = 128,
) -> tuple[list[slice], list[slice], list[slice]]:
    ext_slices = []
    ext_slices_bound = []
    int_slices = []

    # -bound is needed to avoid to incorporate the last chunk, if smaller of bound in the previous chunk
    if dim_size > bound:
        number_of_chunks = int(math.ceil((dim_size - bound) / chunks))
    else:
        number_of_chunks = 1
    for n in range(number_of_chunks):
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
    slices: list[list[slice]], dims_name: list[str]
) -> list[dict[str, slice]]:
    product: list[dict[str, slice]] = []

    for slices_ in itertools.product(*slices):
        product.append({})
        for dim, sl in zip(dims_name, slices_):
            product[-1][dim] = sl
    return product


def compute_chunks(
    dims: dict[str, int] = {},
    chunks: int = 2048,
    bound: int = 128,
) -> tuple[list[dict[str, slice]], list[dict[str, slice]], list[dict[str, slice]]]:
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


def map_ovelap(
    function: Callable[..., xr.DataArray],
    obj: xr.Dataset | xr.DataArray,
    chunks: int = 2048,
    bound: int = 128,
    kwargs: dict[Any, Any] = {},
    template: xr.DataArray | None = None,
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

    try:
        from dask.array import empty_like
    except ModuleNotFoundError:
        from numpy import empty_like  # type: ignore

    out = xr.DataArray(empty_like(template.data), dims=template.dims)  # type: ignore
    out.coords.update(obj.coords)
    for ext_chunk, ext_chunk_bounds, int_chunk in zip(
        ext_chunks, ext_chunks_bounds, int_chunks
    ):
        out_chunk = function(obj.isel(ext_chunk), **kwargs)
        out[int_chunk] = out_chunk.isel(ext_chunk_bounds)
    return out
