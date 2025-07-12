import numpy as np
import pandas as pd 
from Util.Nifti import NiftiHandler
import tqdm 
import nibabel as nib

class Model:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._nifti_handler = NiftiHandler()

    @property
    def df(self) -> pd.DataFrame:
        '''read only access'''
        return self._df 
    
    @property
    def nifti_handler(self) -> NiftiHandler:
        '''read only'''
        return self._nifti_handler
    
    def shuffled_df(self, random_state: int = 42) -> pd.DataFrame:
        """Shuffles dataframe with a random seed"""
        return self.df.sample(
            frac = 1,
            random_state = random_state
        )
    
    def get_split_idx(self) -> int:
        '''returns the index where current df is split into 
        2/3rds training 1/3rd validating'''
        return len(self.df) * 2 // 3
    
    def split_data(self, data: pd.DataFrame | pd.Series) -> tuple[pd.DataFrame | pd.Series]:
        """Splits data into training and validating data

        Args:
            data (pd.DataFrame | pd.Series): Data to be split

        Returns:
            tuple[pd.DataFrame | pd.Series]: Training Data, Validating Data
        """
        split_idx = self.get_split_idx()
        return data.iloc[:split_idx], data.iloc[split_idx:]

    def create_training_data(self, ROI: str = 'Prefontal Cortex') -> tuple[np.array]:
        """Creates the training/validating X & Y data

        Args:
            ROI (str, optional): Model is trained on different ROIs (region of interest) of the brain. Defaults to 'Prefontal Cortex'.

        Returns:
            tuple[np.array]: X training, X validating, Y training, Y validating
        """
        df = self.shuffled_df()

        Y = df[['Age']]

        X_splits = X_tr, X_v = [], []
        Y_splits = Y_tr, Y_v = self.split_data(Y)
        tqdm_descriptions = ['Training', 'Validating']

        for X_split, Y_split, desc in zip(X_splits, Y_splits, tqdm_descriptions):
            for patient_id in tqdm(Y_split.index, desc=f'Making {desc} Data Set'):

                self.nifti_handler.img = nib.load(f'./Data/func/{patient_id}.nii.gz')
                roi_img = self.nifti_handler.get_roi_img(ROI)
                roi_data = roi_img.get_fdata()

                X_split.append(roi_data)
        
        return X_tr, X_v, Y_tr, Y_v