# Requirements

Requirements status: :white_check_mark: for final and :construction: for draft

## General

1. :white_check_mark: _Package type_: Command line tool and python library

1. :white_check_mark: _License_: Apache v2

1. :white_check_mark: _Technology foundation_:
   Python, xarray / dask and rioxarray / rasterio / GDAL / libgeotiff / PROJ

1. :white_check_mark: _Target platform_: Ubuntu 20.04 LTS (No additional limitation on top of the technology foundation)

1. :white_check_mark: _Home repository_: On the GitHub "bopen" organisation

1. :white_check_mark: _Distribution channels_: PyPI and conda-forge

## Input / output

7. :white_check_mark: _Input DEM formats_: Any file supported by GDAL / PROJ

1. :white_check_mark: _Input SAR products_: Sentinel-1 L1 GRD

1. :white_check_mark: _Input access_:
   Shall be able to load scenes from the filesystem or from the Sentinel-1 GRD archive on Azure Blob Storage

1. :white_check_mark: _Output format_: Cloud Optimized GeoTIFF plus the associated STAC Item

## Functionalities

11. :white_check_mark: _Geometric terrain correction_:
    Shall apply geometric terrain corrections based on
    [D. Small et al, "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019](https://sentinel.esa.int/documents/247904/1653442/Guide-to-Sentinel-1-Geocoding.pdf)

01. :white_check_mark: _Internal geometric accuracy_:
    Interferometric accuracy (Can be tested on SLC products by comparing the coherence values with the ones obtained with SNAP)

01. :white_check_mark: _Geometric accuracy:_
    Comparable with the accuracy of Sentinel-1 RTC products on Azure Blob Storage

01. :white_check_mark: _Radiometric terrain correction_:
    Shall apply radiometric terrain corrections based on
    [D. Small, "Flattening Gamma: Radiometric Terrain Correction for SAR Imagery" in IEEE Transactions on Geoscience and Remote Sensing, vol. 49, no. 8, pp. 3081-3093, Aug. 2011, doi: 10.1109/TGRS.2011.2120616](https://www.doi.org/10.1109/TGRS.2011.2120616)

01. :white_check_mark: _Radiometric accuracy_:
    Comparable with the accuracy of Sentinel-1 RTC products on Azure Blob Storage (Acceptance criteria/process to be defined)

## Efficiency

16. :white_check_mark: _Chunked data access_:
    Data access shall use chunking for all potentially large input files, in particular: SAR imagery and DEM files

01. :white_check_mark: _Chunked processing_:
    Performance critical algorithms shall be able to perform processing on chunks of input data

01. :construction: _Parallel processing_:
    Applying the terrain correction on a single image shall scale with the number of CPU cores assigned to the computation
