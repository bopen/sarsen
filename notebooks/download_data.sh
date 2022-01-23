
mkdir -p data

# Chicago SM EPSG:32616 5x5

[ ! -f data/Chicago-5m-DEM.tif ] && \
  eio clip -o data/Chicago-30m-DEM.tif --bounds -88.17 41.91 -87.92 42.08 &&
  gdalwarp -r bilinear -s_srs EPSG:4326+5773 -t_srs EPSG:32616 -tr 5 5 -ot Float32 \
    data/Chicago-30m-DEM.tif data/Chicago-5m-DEM.tif

## SLC
sentinelsat --path data -d --include-pattern "*vv*" \
  --name S1B_S6_SLC__1SDV_20211216T115438_20211216T115501_030050_03968A_4DCB

## GRD
sentinelsat --path data -d --include-pattern "*vv*" \
  --name S1B_S6_GRDH_1SDV_20211216T115438_20211216T115501_030050_03968A_0F8A

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
