"""
Danny Salib
07/11/2025

Python 3.11.9

"""
import logging
import warnings
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
import nibabel as nib
from tqdm import tqdm
from Util.Nifti.NiftiHandler import NiftiHandler
from download_data import  data_path, download_data

warnings.filterwarnings('ignore')

logging.basicConfig(filename="report.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()

def print_class_prior(df: pd.DataFrame) -> None:
    """
    prints class prior report on data frame
    """
    classification_group_names = ['AgeGroup', 'Child_Adult', 'Gender']

    for class_num, name in enumerate(classification_group_names, start=1):
        print(f'##### Classification {class_num}: {name}')

        for classification_group_name, df_classification_group in df.groupby(name):
            p = len(df_classification_group) / len(df)
            class_prior = f'- $P(Y={classification_group_name}) = {p * 100:.2f}\%$'

            logger.info(class_prior)
            print(class_prior)

    print(f'##### n = {len(df)}')

def run_report():
    """
    Runs model and print a mark down report to STDO
    """
    print('# Demographic Predictive Modeling from MRIs')
    print('##### Predicting age/gender from structural MRI (T1) using CNNs')

    if not os.path.exists(data_path):
        logger.info('%s doesnt exist.', data_path)
        logger.info('Downloading data...')
        download_data()

    print('### Pre-processed Data')
    print('---')

    df_path = r'Data\participants.tsv'

    try:
        df = pd.read_csv(df_path, sep='\t')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'{e}\nTry running `download_data.py` first!') from e

    df = df.set_index('participant_id')

    print(df.head().to_markdown())
    print('#### Class Prioirs')
    print('---')
    print_class_prior(df)
    print('#### Class Priors After Resampling')

    logger.info('Resampling...')

    minority_age_group_count = df['AgeGroup'].value_counts().min()

    # Balance AgeGroup
    df = pd.concat([
        resample(df[df['AgeGroup'] == age],
                replace=True,
                n_samples=minority_age_group_count,
                random_state=42)
        for age in df['AgeGroup'].unique()
    ])

    print_class_prior(df)

    logger.info('Making training data...')

    df = df.sample(frac=1) # shuffle df
    #split_idx = len(df) * 2 // 3 # split 2/3 training & 1/3
    Y = df['Age']
    training_data = [] # A 5D array. will need to use PCA (principle component analysis)
    nifti_handler = NiftiHandler()

    for patient_id in tqdm(Y.index, desc='Making ML Data Set'):
        img = nib.load(f'./Data/func/{patient_id}.nii.gz')
        nifti_handler.img = img
        pfc_img = nifti_handler.get_roi_img('Prefrontal Cortex')
        pfc_data = pfc_img.get_fdata()

        training_data.append(pfc_data)

    logger.info('Running PCA analysis...')

    # Reshape each 5D array to 2D and concatenate
    training_data = np.vstack([
        fMRI.reshape(-1, fMRI.shape[-2] * fMRI.shape[-1])  # Flatten spatial dims, keep time×condition as samples
        for fMRI in training_data
    ])

    # Remove background voxels (all zeros) to save memory/compute:
    training_data = training_data[:, ~np.all(training_data == 0, axis=0)]

    scaler = StandardScaler()
    training_data = scaler.fit_transform(training_data)  # Z-score each voxel

    pca = PCA(n_components=50)
    training_data = pca.fit_transform(training_data)

    print(training_data)
    logger.info(training_data)


    logger.info('finished')
if __name__ == '__main__':
    run_report()