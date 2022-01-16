import os

import sentinelsat

data = [
    ("*iw3*vv*", "S1B_IW_SLC__1SDV_20211223T051121_20211223T051148_030148_039993_BA4B"),
    ("*iw3*vv*", "S1A_IW_SLC__1SDV_20211229T051151_20211229T051218_041219_04E60D_F691"),
    ("*vv*", "S1A_S6_SLC__1SDV_20211222T115519_20211222T115543_041121_04E2C7_EAA4"),
    ("*vv*", "S1B_S6_SLC__1SDV_20211216T115438_20211216T115501_030050_03968A_4DCB"),
    ("*vv*", "S1A_S6_SLC__1SDV_20211210T115520_20211210T115543_040946_04DCEB_0E6D"),
]


user = os.environ["DHUS_USER"]
password = os.environ["DHUS_PASSWORD"]
here = os.path.dirname(__file__)
output_folder = os.path.join(here, "data")

api = sentinelsat.SentinelAPI(user, password)

for filter, identifier in data:
    nodefilter = sentinelsat.make_path_filter(filter)
    s1b_rome = api.query(identifier=identifier)
    api.download_all(s1b_rome, output_folder, nodefilter=nodefilter, n_concurrent_dl=1, fail_fast=True)


# import elevation
# elevation.clip(bounds=(12.40, 41.91, 12.65, 42.08), output=f"{output_folder}/Rome-30m-DEM.tif")
# elevation.clip(bounds=(-88.17, 41.91, -87.92, 42.08), output=f"{output_folder}/Chicago-30m-DEM.tif")
