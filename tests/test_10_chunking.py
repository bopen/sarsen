import numpy as np
import xarray as xr

from sarsen import chunking


def test_compute_chunk_1d() -> None:
    ext_chunks, ext_chunks_bound, int_chunks = chunking.compute_chunks_1d(
        dim_size=20, chunks=10, bound=2
    )
    assert ext_chunks == [slice(0, 12), slice(8, 20)]
    assert int_chunks == [slice(0, 10), slice(10, 20)]
    assert ext_chunks_bound == [slice(0, 10), slice(2, 12)]

    ext_chunks, ext_chunks_bound, int_chunks = chunking.compute_chunks_1d(
        dim_size=13, chunks=10, bound=2
    )
    assert ext_chunks == [slice(0, 12), slice(8, 13)]
    assert int_chunks == [slice(0, 10), slice(10, 13)]
    assert ext_chunks_bound == [slice(0, 10), slice(2, 5)]

    # check bound case: k * chunks + bound == dim_size
    ext_chunks, ext_chunks_bound, int_chunks = chunking.compute_chunks_1d(
        dim_size=22, chunks=10, bound=2
    )
    assert ext_chunks == [slice(0, 12), slice(8, 22)]
    assert int_chunks == [slice(0, 10), slice(10, 22)]
    assert ext_chunks_bound == [slice(0, 10), slice(2, 14)]

    # check bound case: k * chunks + bound > dim_size >  k * chunks
    ext_chunks, ext_chunks_bound, int_chunks = chunking.compute_chunks_1d(
        dim_size=21, chunks=10, bound=2
    )
    assert ext_chunks == [slice(0, 12), slice(8, 21)]
    assert int_chunks == [slice(0, 10), slice(10, 21)]
    assert ext_chunks_bound == [slice(0, 10), slice(2, 13)]

    # check bound case: dim_size < bound
    ext_chunks, ext_chunks_bound, int_chunks = chunking.compute_chunks_1d(
        dim_size=2, chunks=10, bound=10
    )
    assert ext_chunks == [slice(0, 2)]
    assert int_chunks == [slice(0, 2)]
    assert ext_chunks_bound == [slice(0, 2)]


def test_compute_chunks() -> None:
    ext_chunks, ext_chunks_bound, int_chunks = chunking.compute_chunks(
        dims={"x": 10, "y": 21}, chunks=10, bound=2
    )
    assert {"x": slice(0, 10), "y": slice(0, 12)} in ext_chunks
    assert {"x": slice(0, 10), "y": slice(8, 21)} in ext_chunks

    assert {"x": slice(0, 10), "y": slice(0, 10)} in int_chunks
    assert {"x": slice(0, 10), "y": slice(10, 21)} in int_chunks

    assert {"x": slice(0, 10), "y": slice(0, 10)} in ext_chunks_bound
    assert {"x": slice(0, 10), "y": slice(2, 13)} in ext_chunks_bound


def test_map_ovelap() -> None:
    arr = xr.DataArray(np.arange(22 * 31).reshape((22, 31)), dims=("x", "y"))

    def function(x: xr.DataArray) -> xr.DataArray:
        return x

    res = chunking.map_ovelap(function=function, obj=arr, chunks=10, bound=2)
    assert res.equals(arr)
