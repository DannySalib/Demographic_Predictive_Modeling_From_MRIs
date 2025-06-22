import boto3
from botocore import UNSIGNED
from botocore.client import Config
import shutil 
import os 

'''
Constants and Helper Functions
______________________________
'''
# Info on the data set
__DATA_SET_NAME = 'ds000228'
__BUCKET = 'openneuro.org' # for s3 requests (bucket refers to the name of the website)
__DATA_SET_URL = f'https://{__BUCKET}/datasets/{__DATA_SET_NAME}'
__DATA_PATH = '../Data' # Create destination folder for data

# To get an idea of the naming conventions, here is an example key you'd use to download a file from the dataset
example_key = 'ds000228/sub-pixar001/anat/sub-pixar001_T1w.nii.gz'
example_file_name = 'sub-pixar001_T1w.nii.gz'

# Validates the key when making requests to openneuro.org 
is_valid_key = lambda key: key.startswith('ds000228') # lazy checking 
def validate_key(func):
    def wrapper(*args, **kwargs):
        key = kwargs.get('key')
        
        if key and not is_valid_key(key):
            raise Exception(
                f'Please select a file from {__DATA_SET_URL}\n'
                f'Example key: \"{example_key}\"'
            )
        
        return func(*args, **kwargs)

    return wrapper

'''
Downloading Data
________________
'''
# Set up s3 client 
__s3_client = boto3.client(
    's3',
    endpoint_url='https://s3.amazonaws.com',
    config=Config(
        signature_version=UNSIGNED
        )
)


@validate_key
def download_file(key: str, the_file_name_you_want: str) -> str:
    '''
    Downloads file from https://openneuro.org/datasets/ds000228 and moves it to the Data directory
    Returns a str path describing file path
    '''
    output_path = f'{__DATA_PATH}/{the_file_name_you_want}'
    __s3_client.download_file(
        Bucket= __BUCKET,
        Key=key,  # Use the exact path from list_objects_v2
        Filename= output_path
    )

    return f'Downloaded to {output_path}'


