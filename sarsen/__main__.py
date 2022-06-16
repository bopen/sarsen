import json
import typing as T

import typer

from . import apps

app = typer.Typer()


@app.command()
def info(
    product_urlpath: str,
) -> None:
    """Print information about the Sentinel-1 product."""
    product_info = apps.product_info(product_urlpath)
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
    chunks: T.Optional[int] = 1024,
) -> None:
    """Generate a geometrically terrain corrected (GTC) image from Sentinel-1 product."""
    client_kwargs = json.loads(client_kwargs_json)
    chunks = chunks if chunks > 0 else None
    apps.terrain_correction(
        product_urlpath,
        measurement_group,
        dem_urlpath,
        output_urlpath=output_urlpath,
        enable_dask_distributed=enable_dask_distributed,
        client_kwargs=client_kwargs,
        chunks=chunks,
    )


@app.command()
def rtc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str = "RTC.tif",
    enable_dask_distributed: bool = False,
    client_kwargs_json: str = '{"processes": false}',
    chunks: T.Optional[int] = 0,
    grouping_area_factor: T.Tuple[float, float] = (3.0, 3.0),
) -> None:
    """Generate a radiometrically terrain corrected (RTC) image from Sentinel-1 product."""
    client_kwargs = json.loads(client_kwargs_json)
    chunks = chunks if chunks > 0 else None
    apps.terrain_correction(
        product_urlpath,
        measurement_group,
        dem_urlpath,
        correct_radiometry="gamma_nearest",
        output_urlpath=output_urlpath,
        grouping_area_factor=grouping_area_factor,
        enable_dask_distributed=enable_dask_distributed,
        client_kwargs=client_kwargs,
        chunks=chunks,
    )


if __name__ == "__main__":
    app()
