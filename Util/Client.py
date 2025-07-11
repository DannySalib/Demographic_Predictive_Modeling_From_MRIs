"""
Danny Salib
07/11/2025
client.py = handles boto3 client to get all relevent data for this project
Python 3.11.9
"""

import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config

########################## Constants and Helper Functions
# Info on the data set
__DATA_SET_NAME = 'ds000228'
__BUCKET = 'openneuro.org' # for s3 requests (bucket refers to the name of the website)
__DATA_SET_URL = f'https://{__BUCKET}/datasets/{__DATA_SET_NAME}'
__DATA_PATH = '../Data' # Create destination folder for data

@lambda _:_() # i get to do wacky shit like this cuz its a solo project
def data_path() -> str:
    '''read only kinda lke python's  @property built in'''
    return __DATA_PATH

# To get an idea of the naming conventions,
#   here is an example key you'd use to download a file from the dataset
EXAMPLE_KEY = 'ds000228/sub-pixar001/anat/sub-pixar001_T1w.nii.gz'
EXAMPLE_FILE_NAME = 'sub-pixar001_T1w.nii.gz'

################# Downloading Data

# Set up s3 client
__s3_client = boto3.client(
    's3',
    endpoint_url='https://s3.amazonaws.com',
    config=Config(
        signature_version=UNSIGNED
        )
)

def download_file(key: str, the_file_name_you_want: str, path: str = '.') -> str:
    '''
    Downloads file from https://openneuro.org/datasets/ds000228 and moves it to the Data directory
    Returns a str path describing file path
    '''
    os.makedirs(path, exist_ok=True)
    output_path = f'{path}/{the_file_name_you_want}'
    __s3_client.download_file(
        Bucket= __BUCKET,
        Key=key,  # Use the exact path from list_objects_v2
        Filename= output_path
    )

    return f'Downloaded to {output_path}'