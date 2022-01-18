# Sarsen

Algorithms and utilities for SAR sensors. Enables cloud-native SAR processing via *Xarray* and *Dask*.

## Target features

Overall the software is in early **alpha** phase and the usual caveats apply.

- provides algorithms to terrain-correct satellite SAR data
  - geometric terrain correction (geocoding)
    - *fast mode*: to terrain-correct images
    - *accurate mode*: for interferometric processing
  - radiometric terrain correction (gamma flattening)
- SAR data access via [*xarray-sentinel*](https://github.com/bopen/xarray-sentinel):
  - supports all Sentinel-1 data products as [distributed by ESA](https://scihub.copernicus.eu/dhus/#/home):
    - Sentinel-1 Single Look Complex (SLC) SM/IW/EW/WV
    - Sentinel-1 Ground Range Detected (GRD) SM/IW/EW/WV
  - reads uncompressed and compressed SAFE data products on the local computer or
    on a network via [*fsspec*](https://filesystem-spec.readthedocs.io)
- DEM data access via [*rioxarray*](https://corteva.github.io/rioxarray):
  - reads local and remote data in virtually any raster format via
    [*rasterio*](https://rasterio.readthedocs.io) / [*GDAL*](https://gdal.org)
- supports larger-than-memory and distributed data access and processing via [*dask*](https://dask.org)

## Non-objectives / Caveat emptor

- No attempt is done to support UTC leap seconds. Observations that include a leap second may crash the code or
  silently return wrong results.

## Usage

```python-repl
>>> import sarsen

```

## Project resources

[![on-push](https://github.com/bopen/sarsen/actions/workflows/on-push.yml/badge.svg)](https://github.com/bopen/sarsen/actions/workflows/on-push.yml)
[![codecov](https://codecov.io/gh/bopen/sarsen/branch/main/graph/badge.svg?token=62S9EXDF0V)](https://codecov.io/gh/bopen/sarsen)

## Contributing

The main repository is hosted on GitHub,
testing, bug reports and contributions are highly welcomed and appreciated:

https://github.com/bopen/sarsen

Lead developer:

- [Alessandro Amici](https://github.com/alexamici) - [B-Open](https://bopen.eu)

Main contributors:

- [Aureliana Barghini](https://github.com/aurghs) - [B-Open](https://bopen.eu)

See also the list of [contributors](https://github.com/bopen/sarsen/contributors) who participated in this project.

## Sponsoring

[B-Open](https://bopen.eu) commits to maintain the project long term and we are happy to accept sponsorships to develop new features.

We wish to express our gratitude to the project sponsors:

- [Microsoft](https://microsoft.com)

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
