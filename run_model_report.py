import pandas as pd 
import numpy as np 
from sklearn.utils import resample
import nibabel as nib
from Util.Nifti.NiftiHandler import NiftiHandler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def main():
    print('# Demographic Predictive Modeling from MRIs')
    print('##### Predicting age/gender from structural MRI (T1) using CNNs')
    print('### Pre-processed Data')
    print('---')
    df_path = r'Data\participants.tsv'
    try:
        df = pd.read_csv(df_path, sep='\t')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'{e}\nTry running `download_data.py` first!')
    df = df.set_index('participant_id')
    print(df.head().to_markdown())
    print('#### Class Prioirs')
    print('---')
    classification_group_names = ['AgeGroup', 'Child_Adult', 'Gender']
    for class_num, name in enumerate(classification_group_names, start=1):
        print(f'##### Classification {class_num}: {name}')
        for classification_group_name, df_classification_group in df.groupby(name):
            p = len(df_classification_group) / len(df)
            print(f'- $P(Y={classification_group_name}) = {p * 100:.2f}\%$')
    print(f'##### n = {len(df)}')
    print('#### Class Priors After Resampling')
    minority_age_group_count = df['AgeGroup'].value_counts().min()
    # Step 1: Balance AgeGroup
    df = pd.concat([
        resample(df[df['AgeGroup'] == age], 
                replace=True, 
                n_samples=minority_age_group_count, 
                random_state=42)
        for age in df['AgeGroup'].unique()
    ])
    classification_group_names = ['AgeGroup', 'Child_Adult', 'Gender']
    for class_num, name in enumerate(classification_group_names, start=1):
        print(f'##### Classification {class_num}: {name}')
        for classification_group_name, df_classification_group in df.groupby(name):
            p = len(df_classification_group) / len(df)
            print(f'- $P(Y={classification_group_name}) = {p * 100:.2f}\%$')
    print(f'##### n = {len(df)}')
    df = df.sample(frac=1) # shuffle df 
    split_idx = len(df) * 2 // 3 # split 2/3 training & 1/3
    Y = df['Age']
    X = []
    nifti_handler = NiftiHandler()
    for id in tqdm(Y.index, desc='Making ML Data Set'):
        img = nib.load(f'./Data/func/{id}.nii.gz')
        nifti_handler.img = img

        pfc_img = nifti_handler.get_roi_img('Prefrontal Cortex')
        pfc_data = pfc_img.get_fdata()
        X.append(pfc_data)

if __name__ == '__main__':
    main()