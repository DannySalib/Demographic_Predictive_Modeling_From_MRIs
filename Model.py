import os
import warnings
import numpy as np
import pandas as pd 
from Util.Nifti.NiftiHandler import NiftiHandler
import nibabel as nib
from multiprocessing import Pool, cpu_count
from typing import Generator

PFC_ROI_STR: str = 'Prefrontal Cortex'
MIN_PROCESSOR_CORES: int = 4
class Model:
    def __init__(self, participants_df: pd.DataFrame):
        self._participants_df = participants_df
        self._nifti_handler = NiftiHandler()

    @property
    def participants_df(self) -> pd.DataFrame:
        '''read only access'''
        return self._participants_df 
    
    @property
    def nifti_handler(self) -> NiftiHandler:
        '''read only'''
        return self._nifti_handler
    
    def shuffled_df(self, random_state: int = 42) -> pd.DataFrame:
        """Shuffles dataframe with a random seed"""
        return self.participants_df.sample(
            frac = 1,
            random_state = random_state
        )
    
    def get_split_idx(self) -> int:
        '''returns the index where current df is split into 
        2/3rds training 1/3rd validating'''
        return len(self.participants_df) * 2 // 3
    
    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame]:
        """Splits data into training and validating data

        Args:
            data (pd.DataFrame): Data to be split

        Returns:
            tuple[pd.DataFrame]: Training Data, Validating Data
        """
        split_idx = self.get_split_idx()
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    def _load_single_ROI(self, patient_id: str, ROI: str) -> np.array:
        """Try to load the patients ROI 

        Args:
            patient_id (str): Patient ID
            ROI (str): Region of interest 

        Returns:
            np.array | None: img data if successfull. None otherwise 
        """
        img_path = f'../Data/func/{patient_id}.nii.gz'
        
        self.nifti_handler.img = nib.load(img_path)
        
        roi_img = self.nifti_handler.get_roi_img(ROI)

        roi_data = roi_img.get_fdata() # should be 4D fMRI data

        return roi_data.flatten()
        
    def _load_ROIs(self, patient_ids: pd.Index, ROI: str,  n_workers: int = None) -> Generator:
        """Load the ROI of each patient in parallel and yield each patient ROI

        Args:
            patient_ids (pd.Indexn): Patient ID
            ROI (str): Region of interest 
            n_workers (int, optional): Number of cores to utilize in parallel processing. Defaults to None.

        Yields:
            Generator: yield of ROI data
        """
        # TODO to reduce filler 0s (background voxels) get the ROI mask and use to to drop uneeded 0s
        n_workers = n_workers or min(MIN_PROCESSOR_CORES, cpu_count() - 1)
        with Pool(n_workers) as pool:
            all_roi_data = pool.starmap(
                self._load_single_ROI,
                [(pid, ROI) for pid in patient_ids]
            )

            for roi_data in all_roi_data:
                yield roi_data

    def create_training_data(self, ROI: str = PFC_ROI_STR, n_workers: int = None) -> tuple[Generator, Generator, np.array, np.array]:
        """Creates the training/validating X & Y data

        Args:
            ROI (str, optional): Model is trained on different ROIs (region of interest) of the brain. Defaults to 'Prefontal Cortex'.

        Returns:
            tuple[np.array]: X training, X validating, Y training, Y validating
        """
        df = self.shuffled_df()

        Y = df[['Age']]
        Y_tr, Y_v = self.split_data(Y)

        X_tr_gen = self._load_ROIs(
            patient_ids = Y_tr.index,
            ROI = ROI,
            n_workers = n_workers
        ) 

        X_v_gen = self._load_ROIs(
            patient_ids = Y_v.index,
            ROI = ROI,
            n_workers = n_workers
        )

        return X_tr_gen, X_v_gen, Y_tr, Y_v