"""
Danny Salib
07/11/2025
download_data.py = accesses opennuero.org's ds000228 dataset and downloads it
Python 3.11.9

"""
import os
import pandas as pd
from tqdm import tqdm
import Util.client as Client

__DATA_PATH= './Data'

@lambda _:_() # its a solo project so i get to do wacky shi like this
def data_path() -> str:
    '''read only'''
    return __DATA_PATH

def download_data():
    """
    Iterates through each patient id to get func and anat data
    """
    # Create data folder
    os.makedirs(__DATA_PATH, exist_ok=True)

    # Add anat data folder. this is where the patient's base brain state goes
    anat_path = f'{__DATA_PATH}/anat'
    os.makedirs(anat_path, exist_ok=True)

    # add func data folder. this is where the patients's brain activity during task goes
    func_path = f'{__DATA_PATH}/func'
    os.makedirs(func_path, exist_ok=True)

    # get the particpants data frame
    participants_df_key = 'ds000228/participants.tsv'
    participants_df_path = f'{__DATA_PATH}/participants.tsv'
    Client.download_file(
        key = participants_df_key,
        the_file_name_you_want = participants_df_path
    )

    try:
        participants_df = pd.read_csv(participants_df_path, sep='\t')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Could not download file: {participants_df_key}') from e


    for participant_id in tqdm(participants_df['participant_id'], desc='Collecting Data'):

        anat_key = f'ds000228/{participant_id}/anat/{participant_id}_T1w.nii.gz'
        anat_name = f'{anat_path}/{participant_id}.nii.gz'
        Client.download_file(anat_key, anat_name)

        func_key = f'ds000228/{participant_id}/func/{participant_id}_task-pixar_bold.nii.gz'
        func_name = f'{func_path}/{participant_id}.nii.gz'
        Client.download_file(func_key, func_name)

if __name__ == '__main__':
    download_data()