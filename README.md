# Sarsen

Algorithms and utilities for Synthetic Aperture Radar (SAR) sensors.
Enables cloud-native SAR processing via [*Xarray*](https://xarray.pydata.org)
and [*Dask*](https://dask.org).

This Open Source project is sponsored by B-Open - https://www.bopen.eu.

## Features and limitations

*Sarsen* is a Python library and command line tool with the following functionalities:

- provides algorithms to terrain-correct satellite SAR data
  - geometric terrain correction (geocoding)
    - *fast mode*: to terrain-correct images
    - *accurate mode*: for interferometric processing
  - radiometric terrain correction (gamma flattening)
- accesses SAR data via [*xarray-sentinel*](https://github.com/bopen/xarray-sentinel):
  - supports most Sentinel-1 data products as [distributed by ESA](https://scihub.copernicus.eu/dhus/#/home):
    - Sentinel-1 Single Look Complex (SLC) SM/IW/EW
    - Sentinel-1 Ground Range Detected (GRD) SM/IW/EW
  - reads uncompressed and compressed SAFE data products on the local computer or
    on a network via [*fsspec*](https://filesystem-spec.readthedocs.io) - *depends on rasterio>=1.3*
- accesses DEM data via [*rioxarray*](https://corteva.github.io/rioxarray):
  - reads local and remote data in virtually any raster format via
    [*rasterio*](https://rasterio.readthedocs.io) / [*GDAL*](https://gdal.org)
- supports larger-than-memory and distributed data access and processing via *Dask*
  - efficient geometric terrain-correction for a full GRD
  - efficient radiometric terrain-correction for a full GRD.

Overall, the software is in the **beta** phase and the usual caveats apply.

Current limitations:

- documentation needs improvement. See #6.

Non-objectives / Caveat emptor items:

- No attempt is made to support UTC leap seconds. Observations that include a leap second may
  crash the code or silently return wrong results.

## SAR terrain-correction

The typical side-looking SAR system acquires data with uniform sampling in azimuth and slant range,
where the azimuth and range represents the time when a given target is acquired and the absolute
sensor-to-target distance, respectively.
Because of this, the near range appears compressed with respect to the far range. Furthermore,
any deviation of the target elevation from a smooth geoid results in additional local geometric and radiometric
distortions known as foreshortening, layover and shadow.

- Radar foreshortening: Terrain surfaces sloping towards the radar appear shortened relative to those sloping away from the radar.
  These regions are much brighter than other places on the SAR image.
- Radar layover: It's an extreme case of foreshortening occurring when the terrain slope is greater than the angle of the incident signal.
- Radar shadows: They occur when ground points at the same azimuth but different slant ranges are aligned in the direction of the line-of-sight.
  This is usually due to a back slope with an angle steeper than the viewing angle.
  When this happens, the radar signal never reaches the farthest points, and thus there is no measurement, meaning that this lack of information is unrecoverable.

The geometric terrain correction (GTC) corrects the distortions due to the target elevation.
The radiometric terrain correction (RTC) also compensates for the backscatter modulation generated
by the topography of the scene.

## Install

The easiest way to install *sarsen* is in a *conda* environment.
The following commands create a new environment, activate it, install the package and its dependencies:

```shell
  conda create -n SARSEN
  conda activate SARSEN
  conda install -c conda-forge dask proj-data sarsen
```

Note that the `proj-data` package is rather large (500+Mb) and it is only needed to handle input DEM whose
vertical coordinate is not on a known ellipsoid, for example *SRTM DEM* with heigths over the *EGM96 geoid*.

## Command line usage

The `sarsen` command line tool corrects SAR data based on a selected DEM and may produce
geometrically terrain-corrected images (GTC) or radiometrically terrain-corrected images (RTC).
Terrain-corrected images will have the same pixels as the input DEM, that should be resampled
to the target projection and spacing in advance, for example using
[`gdalwarp`](https://gdal.org/programs/gdalwarp.html).

The following command performs a geometric terrain correction:

```shell
  sarsen gtc S1B_IW_GRDH_1SDV_20211217T141304_20211217T141329_030066_039705_9048.SAFE IW/VV South-of-Redmond-10m_UTM.tif
```

Performing geometric and radiometric terrain correction is more demanding,
but it is possible to produce the RTC of a full GRD product at a 10m resolution
in one go (and it takes approx 25 minutes on a 32Gb MacBook Pro):

```shell
  sarsen rtc S1B_IW_GRDH_1SDV_20211217T141304_20211217T141329_030066_039705_9048.SAFE IW/VV South-of-Redmond-10m_UTM.tif
```

## Python API usage

The python API has entry points to the same commands and it also gives access to several lower level
algorithms, but internal APIs should not be considered stable:

The following code applies the geometric terrain correction to the VV polarization of
"S1B_IW_GRDH_1SDV_20211217T141304_20211217T141329_030066_039705_9048.SAFE" product:

```python
>>> import sarsen
>>> product = sarsen.Sentinel1SarProduct(
...   "tests/data/S1B_IW_GRDH_1SDV_20211223T051122_20211223T051147_030148_039993_5371.SAFE",
...   measurement_group="IW/VV",
... )
>>> gtc = sarsen.terrain_correction(
...   product,
...   dem_urlpath="tests/data/Rome-30m-DEM.tif",
... )

```

The radiometric correction can be activated using the key `correct_radiometry`:

```python
>>> rtc = sarsen.terrain_correction(
...   product,
...   dem_urlpath="tests/data/Rome-30m-DEM.tif",
...   correct_radiometry="gamma_nearest"
... )

```

## Reference documentation

This is the list of the reference documents:

- the geometric terrain-correction algorithms are based on:
  ["Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019](https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3)
- the radiometric terrain-correction algorithms are based on:
  [D. Small, "Flattening Gamma: Radiometric Terrain Correction for SAR Imagery," in IEEE Transactions on Geoscience and Remote Sensing, vol. 49, no. 8, pp. 3081-3093, Aug. 2011, doi: 10.1109/TGRS.2011.2120616](https://www.geo.uzh.ch/microsite/rsl-documents/research/publications/peer-reviewed-articles/201108-TGRS-Small-tcGamma-3809999360/201108-TGRS-Small-tcGamma.pdf)

## Project resources

[![on-push](https://github.com/bopen/sarsen/actions/workflows/on-push.yml/badge.svg)](https://github.com/bopen/sarsen/actions/workflows/on-push.yml)
[![codecov](https://codecov.io/gh/bopen/sarsen/branch/main/graph/badge.svg?token=62S9EXDF0V)](https://codecov.io/gh/bopen/sarsen)

## Contributing

The main repository is hosted on GitHub.
Testing, bug reports and contributions are highly welcomed and appreciated:

https://github.com/bopen/sarsen

Lead developer:

- [Alessandro Amici](https://github.com/alexamici) - [B-Open](https://bopen.eu)

Main contributors:

- [Aureliana Barghini](https://github.com/aurghs) - [B-Open](https://bopen.eu)

See also the list of [contributors](https://github.com/bopen/sarsen/contributors) who participated in this project.

## Sponsoring

[B-Open](https://bopen.eu) commits to maintain the project long term and we are happy to accept sponsorships to develop new features.

We wish to express our gratitude to the project sponsors:

- [Microsoft](https://microsoft.com) has sponsored the support for *GRD* products and the *gamma flattening* algorithm.

## License

```
Copyright 2016-2022 B-Open Solutions srl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
