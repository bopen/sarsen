import json
import logging
from typing import Tuple

import typer

from . import apps, sentinel1

app = typer.Typer()


@app.command()
def info(
    product_urlpath: str,
) -> None:
    """Print information about the Sentinel-1 product."""
    logging.basicConfig(level=logging.INFO)
    product = sentinel1.Sentinel1SarProduct(product_urlpath)
    product_info = product.product_info()
    for key, value in product_info.items():
        print(f"{key}: {value}")


@app.command()
def gtc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str = "GTC.tif",
    enable_dask_distributed: bool = False,
    client_kwargs_json: str = '{"processes": false}',
    chunks: int = 1024,
    seed_step: int | None = None,
) -> None:
    """Generate a geometrically terrain corrected (GTC) image from Sentinel-1 product."""
    client_kwargs = json.loads(client_kwargs_json)
    real_chunks = chunks if chunks > 0 else None
    real_seed_step = (seed_step, seed_step) if seed_step is not None else None
    logging.basicConfig(level=logging.INFO)
    product = sentinel1.Sentinel1SarProduct(
        product_urlpath,
        measurement_group,
    )
    apps.terrain_correction(
        product,
        dem_urlpath,
        output_urlpath=output_urlpath,
        enable_dask_distributed=enable_dask_distributed,
        client_kwargs=client_kwargs,
        chunks=real_chunks,
        seed_step=real_seed_step,
    )


@app.command()
def stc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    simulated_urlpath: str = "STC.tif",
    enable_dask_distributed: bool = False,
    client_kwargs_json: str = '{"processes": false}',
    chunks: int = 1024,
    grouping_area_factor: Tuple[float, float] = (3.0, 3.0),
    seed_step: int | None = None,
) -> None:
    """Generate a simulated terrain corrected image from a Sentinel-1 product."""
    client_kwargs = json.loads(client_kwargs_json)
    real_chunks = chunks if chunks > 0 else None
    real_seed_step = (seed_step, seed_step) if seed_step is not None else None
    logging.basicConfig(level=logging.INFO)
    product = sentinel1.Sentinel1SarProduct(
        product_urlpath,
        measurement_group,
    )
    apps.terrain_correction(
        product,
        dem_urlpath,
        correct_radiometry="gamma_bilinear",
        simulated_urlpath=simulated_urlpath,
        grouping_area_factor=grouping_area_factor,
        enable_dask_distributed=enable_dask_distributed,
        client_kwargs=client_kwargs,
        chunks=real_chunks,
        seed_step=real_seed_step,
    )


@app.command()
def rtc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str = "RTC.tif",
    enable_dask_distributed: bool = False,
    client_kwargs_json: str = '{"processes": false}',
    chunks: int = 1024,
    grouping_area_factor: Tuple[float, float] = (3.0, 3.0),
    seed_step: int | None = None,
) -> None:
    """Generate a radiometrically terrain corrected (RTC) image from Sentinel-1 product."""
    client_kwargs = json.loads(client_kwargs_json)
    real_chunks = chunks if chunks > 0 else None
    real_seed_step = (seed_step, seed_step) if seed_step is not None else None
    logging.basicConfig(level=logging.INFO)
    product = sentinel1.Sentinel1SarProduct(
        product_urlpath,
        measurement_group,
    )
    apps.terrain_correction(
        product,
        dem_urlpath,
        correct_radiometry="gamma_bilinear",
        output_urlpath=output_urlpath,
        grouping_area_factor=grouping_area_factor,
        enable_dask_distributed=enable_dask_distributed,
        client_kwargs=client_kwargs,
        chunks=real_chunks,
        seed_step=real_seed_step,
    )


if __name__ == "__main__":
    app()
