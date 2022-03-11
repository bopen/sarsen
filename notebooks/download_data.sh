
mkdir -p data

# Chicago SM EPSG:32616 4x4

[ ! -f data/Chicago-4m-DEM.tif ] && \
  eio clip -o data/Chicago-30m-DEM.tif --bounds -88.55 41.75 -88.15 42.05 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32616 -tr 4 4 \
      -overwrite -ot Float32 -te 375000 4625000 400000 4650000 \
      -co COMPRESS=DEFLATE data/Chicago-30m-DEM.tif data/Chicago-4m-DEM.tif

# Chicago SM EPSG:32616 10x10

[ ! -f data/Chicago-10m-DEM.tif ] && \
  eio clip -o data/Chicago-30m-DEM.tif --bounds -88.55 41.75 -88.15 42.05 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32616 -tr 10 10 \
      -overwrite -ot Float32 -te 375000 4625000 400000 4650000 \
      -co COMPRESS=DEFLATE data/Chicago-30m-DEM.tif data/Chicago-10m-DEM.tif

## SLC
### descending
sentinelsat --path data -d --include-pattern "*vv*" \
  --name S1B_S6_SLC__1SDV_20211216T115438_20211216T115501_030050_03968A_4DCB

### ascending
sentinelsat --path data -d --include-pattern "*hh*" \
  --name S1A_S4_SLC__1SDH_20211216T234911_20211216T234935_041041_04E01F_428D

## GRD
### descending
sentinelsat --path data -d --include-pattern "*vv*" \
  --name S1B_S6_GRDH_1SDV_20211216T115438_20211216T115501_030050_03968A_0F8A

### ascending
sentinelsat --path data -d --include-pattern "*hh*" \
  --name S1A_S4_GRDH_1SDH_20211216T234911_20211216T234935_041041_04E01F_E341

# Rome IW EPSG:32633 10x10

[ ! -f data/Rome-10m-DEM.tif ] && \
  eio clip -o data/Rome-30m-DEM.tif --bounds 12.2 41.7 12.95 42.25 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32633 -tr 10 10 \
      -overwrite -ot Float32 -te 275000 4625000 325000 4675000 \
      -co COMPRESS=DEFLATE data/Rome-30m-DEM.tif data/Rome-10m-DEM.tif

[ ! -f data/Gran-Sasso-10m-DEM.tif ] && \
  eio clip -o data/Gran-Sasso-30m-DEM.tif --bounds 13.1 42. 13.95 42.75 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32633 -tr 10 10 \
      -overwrite -ot Float32 -te 350000 4660000 400000 4710000 \
      -co COMPRESS=DEFLATE data/Gran-Sasso-30m-DEM.tif data/Gran-Sasso-10m-DEM.tif

[ ! -f data/Gran-Sasso-3m-DEM-small.tif ] && \
  eio clip -o data/Gran-Sasso-30m-DEM.tif --bounds 13.1 42. 13.95 42.75 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32633 -tr 3 3 \
      -overwrite -ot Float32 -te 378000 4700000 383000 4703000 \
      -co COMPRESS=DEFLATE data/Gran-Sasso-30m-DEM.tif data/Gran-Sasso-3m-DEM-small.tif

## SLC
### descending
sentinelsat --path data -d --include-pattern "*iw[23]*vv*" \
  --name S1B_IW_SLC__1SDV_20211223T051121_20211223T051148_030148_039993_BA4B

### ascending
sentinelsat --path data -d --include-pattern "*iw[23]*vv*" \
  --name S1A_IW_SLC__1SDV_20211223T170557_20211223T170624_041139_04E360_B8E2

## GRD
sentinelsat --path data -d --include-pattern "*vv*" \
  --name S1B_IW_GRDH_1SDV_20211223T051122_20211223T051147_030148_039993_5371
