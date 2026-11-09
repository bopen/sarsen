import json
import logging

import click

from . import apps, sentinel1

# the options shared by the terrain correction commands
enable_dask_distributed_option = click.option(
    "--enable-dask-distributed/--no-enable-dask-distributed",
    default=False,
    help="Run on a dask distributed client.",
)
client_kwargs_json_option = click.option(
    "--client-kwargs-json",
    default='{"processes": false}',
    help="Keyword arguments of the dask distributed client, as JSON.",
)
chunks_option = click.option(
    "--chunks",
    default=1024,
    help="Chunk size of the processing, 0 or less for no chunking.",
)
grouping_area_factor_option = click.option(
    "--grouping-area-factor",
    type=(float, float),
    default=(3.0, 3.0),
    help="Size of the radiometric grouping area relative to the product pixel, "
    "in azimuth and range, increase it for low resolution DEMs.",
)
seed_step_option = click.option(
    "--seed-step",
    type=int,
    help="Seed the geocoding with a first pass on every SEED_STEP pixel of the DEM.",
)


@click.group("sarsen", context_settings={"show_default": True})
@click.version_option(package_name="sarsen", prog_name="sarsen")
def app() -> None:
    pass


@app.command()
@click.argument("product_urlpath")
def info(product_urlpath: str) -> None:
    """Print information about a Sentinel-1 product.

    \b
    Arguments:
      PRODUCT_URLPATH  Path or URL of the Sentinel-1 SAFE product.
    """  # noqa: D301 click needs the backspace in "\b" to keep the lines
    logging.basicConfig(level=logging.INFO)
    product = sentinel1.Sentinel1SarProduct(product_urlpath)
    product_info = product.product_info()
    for key, value in product_info.items():
        print(f"{key}: {value}")


@app.command()
@click.argument("product_urlpath")
@click.argument("measurement_group")
@click.argument("dem_urlpath")
@click.option("--output-urlpath", default="GTC.tif", help="Path of the output GTC.")
@enable_dask_distributed_option
@client_kwargs_json_option
@chunks_option
@seed_step_option
def gtc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str,
    enable_dask_distributed: bool,
    client_kwargs_json: str,
    chunks: int,
    seed_step: int | None,
) -> None:
    """Compute the geometric terrain correction.

    Generate a geometrically terrain corrected (GTC) image from a Sentinel-1 product.

    \b
    Arguments:
      PRODUCT_URLPATH    Path or URL of the Sentinel-1 SAFE product.
      MEASUREMENT_GROUP  Measurement group, e.g. IW/VV, see the info command.
      DEM_URLPATH        Path or URL of the DEM, in the output projection and spacing.
    """  # noqa: D301 click needs the backspace in "\b" to keep the lines
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
@click.argument("product_urlpath")
@click.argument("measurement_group")
@click.argument("dem_urlpath")
@click.option(
    "--simulated-urlpath", default="STC.tif", help="Path of the output simulated image."
)
@enable_dask_distributed_option
@client_kwargs_json_option
@chunks_option
@grouping_area_factor_option
@seed_step_option
def stc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    simulated_urlpath: str,
    enable_dask_distributed: bool,
    client_kwargs_json: str,
    chunks: int,
    grouping_area_factor: tuple[float, float],
    seed_step: int | None,
) -> None:
    """Compute the simulated terrain correction.

    Generate a simulated terrain corrected image from a Sentinel-1 product.

    \b
    Arguments:
      PRODUCT_URLPATH    Path or URL of the Sentinel-1 SAFE product.
      MEASUREMENT_GROUP  Measurement group, e.g. IW/VV, see the info command.
      DEM_URLPATH        Path or URL of the DEM, in the output projection and spacing.
    """  # noqa: D301 click needs the backspace in "\b" to keep the lines
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
        output_urlpath=None,
        correct_radiometry="gamma_bilinear",
        simulated_urlpath=simulated_urlpath,
        grouping_area_factor=grouping_area_factor,
        enable_dask_distributed=enable_dask_distributed,
        client_kwargs=client_kwargs,
        chunks=real_chunks,
        seed_step=real_seed_step,
    )


@app.command()
@click.argument("product_urlpath")
@click.argument("measurement_group")
@click.argument("dem_urlpath")
@click.option("--output-urlpath", default="RTC.tif", help="Path of the output RTC.")
@enable_dask_distributed_option
@client_kwargs_json_option
@chunks_option
@grouping_area_factor_option
@seed_step_option
def rtc(
    product_urlpath: str,
    measurement_group: str,
    dem_urlpath: str,
    output_urlpath: str,
    enable_dask_distributed: bool,
    client_kwargs_json: str,
    chunks: int,
    grouping_area_factor: tuple[float, float],
    seed_step: int | None,
) -> None:
    """Compute the radiometric terrain correction.

    Generate a radiometrically terrain corrected (RTC) image from a Sentinel-1 product.

    \b
    Arguments:
      PRODUCT_URLPATH    Path or URL of the Sentinel-1 SAFE product.
      MEASUREMENT_GROUP  Measurement group, e.g. IW/VV, see the info command.
      DEM_URLPATH        Path or URL of the DEM, in the output projection and spacing.
    """  # noqa: D301 click needs the backspace in "\b" to keep the lines
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
