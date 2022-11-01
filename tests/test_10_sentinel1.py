import pathlib

import numpy as np

from sarsen import sentinel1

DATA_FOLDER = pathlib.Path(__file__).parent / "data"

DATA_PATHS = [
    DATA_FOLDER
    / "S1B_IW_GRDH_1SDV_20211223T051122_20211223T051147_030148_039993_5371.SAFE",
    DATA_FOLDER
    / "S1A_IW_SLC__1SDV_20220104T170557_20220104T170624_041314_04E951_F1F1.SAFE",
]


def test_product_info() -> None:
    expected_geospatial_bbox = [
        11.86800305333565,
        40.87886713841886,
        15.32209672548896,
        42.78115380313222,
    ]

    res = sentinel1.product_info(str(DATA_PATHS[0]))

    assert "product_type" in res
    assert np.allclose(res["geospatial_bbox"], expected_geospatial_bbox)
