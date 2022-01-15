import os

# import elevation
import sentinelsat

data = [
    ("*iw3*vv*", "S1B_IW_SLC__1SDV_20211223T051121_20211223T051148_030148_039993_BA4B"),
]


user = os.environ["DHUS_USER"]
password = os.environ["DHUS_PASSWORD"]

api = sentinelsat.SentinelAPI(user, password)

for filter, identifier in data:
    nodefilter = sentinelsat.make_path_filter(filter)
    s1b_rome = api.query(identifier=identifier)
    api.download_all(s1b_rome, "./data", nodefilter=nodefilter)


# elevation.clip(bounds=(12.35, 41.91, 12.65, 42.12), output="./data/Rome-30m-DEM.tif")
