import typer

from . import apps

app = typer.Typer()


@app.command()
def gtc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str = "GTC.tif",
) -> None:
    apps.backward_geocode_sentinel1(
        product_urlpath, measurement_group, dem_urlpath, output_urlpath=output_urlpath
    )


@app.command()
def rtc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str = "RTC.tif",
) -> None:
    apps.backward_geocode_sentinel1(
        product_urlpath,
        measurement_group,
        dem_urlpath,
        output_urlpath=output_urlpath,
        correct_radiometry=True,
    )


if __name__ == "__main__":
    app()
