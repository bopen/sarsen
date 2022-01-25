
mkdir -p data

# Chicago SM EPSG:32616 3.5x3.5

[ ! -f data/Chicago-3.5m-DEM.tif ] && \
  eio clip -o data/Chicago-30m-DEM.tif --bounds -88.5 41.66 -88.2 41.91 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32616 -tr 3.5 3.5 -ot Float32 \
    data/Chicago-30m-DEM.tif data/Chicago-3.5m-DEM.tif

# Chicago SM EPSG:32616 10x10

[ ! -f data/Chicago-10m-DEM.tif ] && \
  eio clip -o data/Chicago-30m-DEM.tif --bounds -88.5 41.66 -88.2 41.91 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32616 -tr 10 10 -ot Float32 \
    data/Chicago-30m-DEM.tif data/Chicago-10m-DEM.tif

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
  eio clip -o data/Rome-30m-DEM.tif --bounds 12.40 41.91 12.65 42.08 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32633 -tr 10 10 -ot Float32 \
    data/Rome-30m-DEM.tif data/Rome-10m-DEM.tif

## SLC
sentinelsat --path data -d --include-pattern "*iw3*vv*" \
  --name S1B_IW_SLC__1SDV_20211223T051121_20211223T051148_030148_039993_BA4B

## GRD
sentinelsat --path data -d --include-pattern "*vv*" \
  --name S1B_IW_GRDH_1SDV_20211223T051122_20211223T051147_030148_039993_5371
