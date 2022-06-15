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
) -> None:
    apps.terrain_correction(
        product_urlpath, measurement_group, dem_urlpath, output_urlpath=output_urlpath
    )


@app.command()
def rtc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str = "RTC.tif",
    grouping_area_factor: T.Tuple[float, float] = (3.0, 13.0),
) -> None:
    apps.terrain_correction(
        product_urlpath,
        measurement_group,
        dem_urlpath,
        correct_radiometry="gamma_nearest",
        output_urlpath=output_urlpath,
        grouping_area_factor=grouping_area_factor,
    )


if __name__ == "__main__":
    app()
