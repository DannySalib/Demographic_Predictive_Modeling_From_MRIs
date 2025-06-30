import pandas as pd
import sys 
import os 
# Add the project root to sys.path
sys.path.append(os.path.abspath('../'))
import Util.Client as Client
from tqdm import tqdm

def main():
    # Create data folder 
    data_path = './Data'
    os.makedirs(data_path, exist_ok=True)

    # Add anat data folder. this is where the patient's base brain state goes 
    anat_path = f'{data_path}/anat'
    os.makedirs(anat_path, exist_ok=True)

    # add func data folder. this is where the patients's brain activity during task goes 
    func_path = f'{data_path}/func'
    os.makedirs(func_path, exist_ok=True)

    # get the particpants data frame 
    participants_df_key = 'ds000228/participants.tsv'
    participants_df_path = f'{data_path}/participants.tsv'
    Client.download_file(
        key = participants_df_key,
        the_file_name_you_want = participants_df_path
    )

    participants_df = pd.read_csv(participants_df_path, sep='\t')

    for participant_id in tqdm(participants_df['participant_id'], desc='Collecting Data'):

        anat_key = f'ds000228/{participant_id}/anat/{participant_id}_T1w.nii.gz'
        anat_name = f'{anat_path}/{participant_id}.nii.gz'
        Client.download_file(anat_key, anat_name)

        func_key = f'ds000228/{participant_id}/func/{participant_id}_task-pixar_bold.nii.gz'
        func_name = f'{func_path}/{participant_id}.nii.gz'
        Client.download_file(func_key, func_name)

if __name__ == '__main__':
    main()