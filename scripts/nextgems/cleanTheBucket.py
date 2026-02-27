import os
import sys
import zarr
import s3fs
import itertools

prodEndpointURL    = 'https://s3.eu-dkrz-1.dkrz.cloud'
preProdEndpointURL = 'https://s3.eu-dkrz-3.dkrz.cloud'


# https://s3.eu-dkrz-1.dkrz.cloud/nextgems/rechunked_ngc4008/ngc4008_{{ time }}_{{ zoom }}.zarr
# Source: https://github.com/digital-earths-global-hackathon/catalog/blob/main/online/main.yaml
globalHackathonPath='nextgems/rechunked_ngc4008' # prodEndpoint

freqList = ['P1D', 'PT3H', 'P15M']
zoomList = [z for z in range(0,11)]

globalHackathonStoreNamesList = [ f'ngc4008_{freq}_{zoom}.zarr' for freq, zoom in itertools.product(freqList, zoomList) ]


def createS3cluster( endPointURL, key, secret ):  
    try:
        if key == '':
            s3Cluster = s3fs.S3FileSystem(
                  anon = True,
                  endpoint_url = endPointURL
            )            
        else:
            s3Cluster = s3fs.S3FileSystem(
                  key = key,
                  secret = secret,
                  endpoint_url = endPointURL
            )           
        return s3Cluster
    except Exception as exp:
        print(f"createS3cluster raises '{exp}' error!!! Check your URL and keys!")        
        sys.exit()


def deleteObjectsFromPath( objPath, s3Cluster ):
    for obj in s3Cluster.ls(objPath) :
        if obj.split('.')[-1] == 'zarr':
            continue
        else:
            print(f"{obj} will be deleted!!!")
            s3Cluster.rm( obj, recursive=True )


def deleteDicretoryFromPath( dirObjPath, s3Cluster ):
    #for obj in s3Cluster.ls(objPath) :
    #    if obj.split('.')[-1] == 'zarr':
    #        continue
    #    else:
    #        print(f"{obj} will be deleted!!!")
    s3Cluster.rm( dirObjPath, recursive=True )




s3PreProd = createS3cluster( preProdEndpointURL,
                             os.getenv("ACCESS_KEY_S3_nextGEMS"),
                             os.getenv("SECRET_KEY_S3_nextGEMS")                
                           )
objPath='nextgems/rechunked_ngc4008/ngc4008_P1D_9.zarr/h*'

deleteDicretoryFromPath( objPath, s3PreProd )