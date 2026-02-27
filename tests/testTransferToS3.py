import xarray as xr
from os import getenv

# Set the appropriate values to the following 
s3PreProdURL = "https://s3.eu-dkrz-3.dkrz.cloud"
destPath     = "nextgems"

data    = xr.DataArray( [ [1,2,3], [4,5,6] ], dims=("x", "y"), coords={ "x": [10, 20] } )
options = {
    "endpoint_url": s3PreProdURL,
    "key"         : getenv("ACCESS_KEY_S3_nextGEMS"),
    "secret"      : getenv("SECRET_KEY_S3_nextGEMS"),
}

data.to_zarr( f's3://{destPath}/dummy.zarr', storage_options = options )
