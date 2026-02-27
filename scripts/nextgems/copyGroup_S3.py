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

'''
# Local directory
base_dir='/home/kamesh/projects/S3Pool/dataset'
sourceStoreName='air_temperature.zarr'


targetPath='nextgems/test'
print( s3PreProd.ls(targetPath) )

# Can be same as source store name or different
targetStoreName=sourceStoreName

# The path including the target zarr store name, where the source zarr store needs to be copied to
targetStorePath=os.path.join(targetPath,targetStoreName)

sourceStore=zarr.DirectoryStore(os.path.join(base_dir,sourceStoreName)) # Source store
targetStore=s3fs.S3Map( root = targetStorePath, s3 = s3PreProd, check = False)  # Target store
'''


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


def copyZarrStoreToTarget( srcStore, tgtStore ):
    zarr.copy_store( srcStore, tgtStore, if_exists='replace' )


def copyZarrGroupToTarget( srcStoreGrp, tgtStoreGrp, tgtStore, grpName ):    
    try:
        zarr.copy( srcStoreGrp[ grpName ],  tgtStoreGrp,  name = grpName, 
                   if_exists = 'replace' # or 'skip', 'raise'
                 )
        # Consolidate the metadata so that '.zmetadata' file is created inside the zarr store
        zarr.consolidate_metadata(tgtStore)
        
    except Exception as excep:
        print(f"\t copyZarrGroupToTarget Raises {excep}")
    
    
# Create s3 clsuters for the endpoints

s3Prod = createS3cluster( prodEndpointURL, '', '')

s3PreProd = createS3cluster( preProdEndpointURL,
                             os.getenv("ACCESS_KEY_S3_nextGEMS"),
                             os.getenv("SECRET_KEY_S3_nextGEMS")                
                           )

'''
# Copy 'Directorystore' on local disk to 'S3store'
copyZarrStoreToTarget( sourceStore, targetStore )


# Generate names for existing stores on a S3 cluster
sourceStoreNamesList=[]
sourcePath='nextgems'
#print( s3PreProd.ls( sourcePath ) )
for obj in s3PreProd.ls( sourcePath ):
    if obj.split('.')[-1] == 'zarr':
        sourceStoreNamesList.append(obj)
print(sourceStoreNamesList)
'''

targetPath='nextgems/test'

# Copy from one zarr store to another zarr store on the same S3 cluster ( here, preProd )
#for srcStoreName in sourceStoreNamesList:

# Copy from one zarr store to another zarr store from different S3 clusters ( here, from prod to preProd )

srcS3Cluster = s3Prod
destS3Cluster = s3PreProd

#print( f" Source cluster object list: {srcS3Cluster.ls( 'wrcp-hackathon/data/contributions/CERES' )}" )
#print( f" Destination cluster object list: {destS3Cluster.ls( targetPath )}" )

for srcStoreName in globalHackathonStoreNamesList[9:10]:
    
    print(  srcStoreName  ) 
    
    # 1. Define Source Store
    print( f"Source store path: {os.path.join( globalHackathonPath, srcStoreName )}" )
    store_source = s3fs.S3Map( root = os.path.join( globalHackathonPath, srcStoreName )  ,
                               s3 = srcS3Cluster, check = False 
                             )  # Source S3 store
    #print( srcS3Cluster.ls( globalHackathonPath  ) )
    #try:
    root_source = zarr.open_group( store=store_source, mode='r' )
    #except Exception as excep:
    #print( f"\tLoop over sourceStoreNamesList Raises {excep}" )
    
    
    # 2. Define Destination Store
    destPath = os.path.join( targetPath, srcStoreName.split(os.path.sep)[-1] )
    if destS3Cluster.exists(destPath):
        deleteObjectsFromPath( destPath, destS3Cluster )
    
    store_dest = s3fs.S3Map( root = destPath, s3 = destS3Cluster, check = False )  # Target S3 store    
    # Create a root for the destination
    root_dest = zarr.open_group(store=store_dest, mode='w')
        
    
    # 3. Copy a specific group from source to destination
    groupNameList = [ 'tas', 'pr' ] #'tas', 'pr'
    for groupName in groupNameList:
        copyZarrGroupToTarget( root_source, root_dest, store_dest, groupName )