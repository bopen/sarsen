import pytest

from sarsen import datamodel


def test_SarProduct() -> None:
    with pytest.raises(TypeError):
        datamodel.SarProduct()  # type: ignore
