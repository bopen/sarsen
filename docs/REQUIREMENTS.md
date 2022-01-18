# Requirements

## General
1. :white_check_mark: _Package type_:	Command line tool and python library
2. :white_check_mark: _License_: Apache v2
3. :white_check_mark: _Techology foundation_:
   - Python
   - xarray / netcdf4
   - dask
   - GDAL / PROJ / rioxarray
4. :white_check_mark: _Target platform_: Ubuntu 20.04 LTS (No additional limitation on top of the technology foundation)
5. :white_check_mark: Home repository	On the GitHub "bopen" organisation
6. :white_check_mark: Distribution channles	PyPI and conda-forge

## Input / output			
7.	Input DEM formats	Any file supported by GDAL / PROJ		Ready
8.	Input SAR products	Sentinel-1 L1 GRD		Ready
9.	Input access	Shall be able to load scenes from filesystem or from the the Sentinel-1 GRD archive on Azure Blob Storage		Ready
10.	Output	GeoTIFF / netCDF4-CF / Zarr	No idea what to do with metadata, though	Draft
				
	Functionalities			
11	Geometric terrain correction	"Shall apply geometric terrain corrections based on: ""Guide to Sentinel-1 Geocoding"" UZH-S1-GC-AD 1.10 26.03.2019
https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3"		Ready
12	Internal geometric accuracy	Interferometric accuracy	Can be tested on SLC products by comparing the coherence values with the ones obtained with GAMMA-RS for example	Draft
13	Geometric accuracy	Compare with GAMMA-RS	"Acceptance criteria/process": do we need to define a validation procedure?	Draft
14	Radiometric terrain correction	Shall be base on the flattening gamma algorithm in: https://ieeexplore.ieee.org/document/5752845		Ready
15	Radiometric accuracy	Compare with GAMMA-RS	"Acceptance criteria/process": do we need to define a validation procedure?	Draft
				
	Efficiency			
16	Chunked data access	"Data access shall use chunking for all potentially large input files, in particular:
- SAR imagery
- DEM"		Draft
17	Chunked processing	Performance critical algorithms shall be optimised to perform processing on chunks of input data		Draft
18	Parallel processing	Applying the terrain correction on a single image shall scale with the number of CPU cores assigned to the computation.		Draft