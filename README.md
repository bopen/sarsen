> **WARNING**: THIS CODE IS NOT READY FOR USE

# Sarsen

Algorithms and utilities for SAR sensors

## Objectives

Be faster and simpler than ESA SNAP and cloud native.

- enable SAR data geocoding
  - *fast mode*: to terrain-correct images
  - *accurate mode*: for interferometric processing
- enable radiometric terrain correction / flattening gamma
- support cloud-native processing
  - enable parallel processing via *xarray* and *dask*
  - enable object storage access via *fsspec*
- support Sentinel-1 SLC IW and GRD
- support any DEM that GDAL / Proj can handle

## Non-objectives

- No attempt is done to support UTC leap seconds. Observations that include a leap second may crash the code or
  silently return wrong results. *Caveat emptor*

## Usage

```python-repl
>>> import sarsen

```
