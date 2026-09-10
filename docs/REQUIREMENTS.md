# Requirements

Requirements status: :white_check_mark: for final and :construction: for draft

## General

1. :white_check_mark: _Package type_: Command line tool and python library

1. :white_check_mark: _License_: Apache v2

1. :white_check_mark: _Technology foundation_:
   Python, xarray / dask and rioxarray / rasterio / GDAL / libgeotiff / PROJ

1. :white_check_mark: _Target platform_: Linux and macOS, with no additional limitation on top of the
   technology foundation. Windows may work but is not tested

1. :white_check_mark: _Home repository_: On the GitHub "bopen" organisation

1. :white_check_mark: _Distribution channels_: PyPI and conda-forge

## Input / output

1. :white_check_mark: _Input DEM formats_: Any file supported by GDAL / PROJ

1. :white_check_mark: _Input SAR products_: Sentinel-1 L1 GRD (SM/IW/EW) and SLC (SM/IW)

1. :white_check_mark: _Input access_:
   Accessing DEMs and images from the filesystem or from a remote object store, e.g. the Copernicus Data Space Ecosystem (`s3://eodata/...`)

1. :white_check_mark: _Output format_: Tiled GeoTIFF, compressed with ZSTD

1. :construction: _Output format, cloud optimisation_: Cloud Optimized GeoTIFF (i.e. with overviews)

1. :construction: _Output metadata_: the STAC Item associated with the output

## Functionalities

1. :white_check_mark: _Geometric terrain correction_:
   Shall apply geometric terrain corrections based on
   [D. Small et al, "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019](https://sentinel.esa.int/documents/247904/1653442/Guide-to-Sentinel-1-Geocoding.pdf)

1. :white_check_mark: _Internal geometric accuracy_: Interferometric accuracy (validated outside of the package with a tighter zero-Doppler tolerance, `zero_doppler_distance=0.001`)

1. :white_check_mark: _Geometric accuracy_:
   Comparable with the accuracy of equivalent Sentinel-1 GTC products

1. :white_check_mark: _Radiometric terrain correction_:
   Applies radiometric terrain corrections based on
   [D. Small, "Flattening Gamma: Radiometric Terrain Correction for SAR Imagery" in IEEE Transactions on Geoscience and Remote Sensing, vol. 49, no. 8, pp. 3081-3093, Aug. 2011, doi: 10.1109/TGRS.2011.2120616](https://www.doi.org/10.1109/TGRS.2011.2120616)

1. :white_check_mark: _Radiometric accuracy_:
   Comparable with the accuracy of equivalent Sentinel-1 RTC products

## Efficiency

1. :white_check_mark: _Chunked data access_:
   Data access uses chunking for all potentially large input files, in particular: SAR imagery and DEM files

1. :white_check_mark: _Chunked processing_:
   Performance critical algorithms are able to perform processing on chunks of input data

1. :white_check_mark: _Parallel processing_:
   Applying the terrain correction on a single image shall scale with the number of CPU cores assigned to the computation
