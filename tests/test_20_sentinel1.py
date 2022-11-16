import pathlib

import numpy as np
import pytest
import xarray as xr

from sarsen import sentinel1

DATA_FOLDER = pathlib.Path(__file__).parent / "data"

DATA_PATHS = [
    DATA_FOLDER
    / "S1B_IW_GRDH_1SDV_20211223T051122_20211223T051147_030148_039993_5371.SAFE",
    DATA_FOLDER
    / "S1A_IW_SLC__1SDV_20220104T170557_20220104T170624_041314_04E951_F1F1.SAFE",
]

GROUPS = ["IW/VV", "IW1/VV"]


@pytest.mark.parametrize("data_path,group", zip(DATA_PATHS, GROUPS))
def test_Sentinel1SarProduct(data_path: str, group: str) -> None:
    res = sentinel1.Sentinel1SarProduct(data_path, group)

    assert isinstance(res.measurement, xr.Dataset)
    assert isinstance(res.orbit, xr.Dataset)
    assert isinstance(res.calibration, xr.Dataset)

    if res.product_type == "GRD":
        assert isinstance(res.coordinate_conversion, xr.Dataset)
        assert res.azimuth_fm_rate is None
        assert res.dc_estimate is None
    else:
        assert res.coordinate_conversion is None
        assert isinstance(res.azimuth_fm_rate, xr.Dataset)
        assert isinstance(res.dc_estimate, xr.Dataset)

    assert res.product_type in {"SLC", "GRD"}
    assert isinstance(res.beta_nought(), xr.DataArray)
    assert isinstance(res.state_vectors(), xr.DataArray)


def test_product_info() -> None:
    expected_geospatial_bbox = [
        11.86800305333565,
        40.87886713841886,
        15.32209672548896,
        42.78115380313222,
    ]
    product = sentinel1.Sentinel1SarProduct(str(DATA_PATHS[0]))

    res = product.product_info()

    assert "product_type" in res
    assert np.allclose(res["geospatial_bbox"], expected_geospatial_bbox)
