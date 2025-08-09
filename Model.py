import os
import warnings
import numpy as np
import pandas as pd
from Util.MRIFunc import MRIFunc

import nibabel as nib
from multiprocessing import Pool, cpu_count
from typing import Generator
from Util.Atlas import Atlas

PFC_ROI_STR: str = 'Prefrontal Cortex'
MIN_PROCESSOR_CORES: int = 4
class Model:
    def __init__(self, participants_df: pd.DataFrame):
        self.participants_df = participants_df
        self.atlas = Atlas()

    def shuffled_df(self, random_state: int = 42) -> pd.DataFrame:
        """Shuffles dataframe with a random seed"""
        return self.participants_df.sample(
            frac = 1,
            random_state = random_state
        )

    def get_split_idx(self) -> int:
        '''returns the index where current df is split into
        2/3rds training 1/3rd validating'''
        nrows, _ = self.participants_df.shape
        return nrows * 2 // 3

    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame]:
        """Splits data into training and validating data

        Args:
            data (pd.DataFrame): Data to be split

        Returns:
            tuple[pd.DataFrame]: Training Data, Validating Data
        """
        split_idx = self.get_split_idx()
        return data.iloc[:split_idx], data.iloc[split_idx:]

    def _load_single_roi(self, patient_id: str, roi: str) -> np.array:
        """Try to load the patients ROI

        Args:
            patient_id (str): Patient ID
            ROI (str): Region of interest

        Returns:
            np.array | None: img data if successfull. None otherwise
        """

        # TODO bad way of determining img_path
        img_path = f'../Data/func/{patient_id}.nii.gz'

        img = nib.load(img_path)
        fMRI = MRIFunc(img.dataobj, img.affine, img.header)
        fMRI_roi_data: np.ndarray = fMRI.get_roi_fdata(roi, preserve_shape=True)

        # remove z-slices that are all 0s
        # TODO
        
        nt = fMRI_roi_data.shape[-1]
        return fMRI_roi_data.reshape(shape=(nt, -1))

    def _load_rois(self, patient_ids: pd.Index, roi: str,  n_workers: int = None) -> Generator:
        """Load the ROI of each patient in parallel and yield each patient ROI

        Args:
            patient_ids (pd.Indexn): Patient ID
            ROI (str): Region of interest
            n_workers (int, optional): Number of cores to utilize in parallel processing. Defaults to None.

        Yields:
            Generator: yield of ROI data
        """
        n_workers = n_workers or min(MIN_PROCESSOR_CORES, cpu_count() - 1)
        with Pool(n_workers) as pool:
            all_roi_data = pool.starmap(
                self._load_single_roi,
                [(pid, roi) for pid in patient_ids]
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

        y = df[['Age']] # TODO this is age, but what if we wanted a different predictor?
        y_tr, y_v = self.split_data(y)

        x_tr_gen = self._load_rois(
            patient_ids = y_tr.index,
            roi = ROI,
            n_workers = n_workers
        )

        x_v_gen = self._load_rois(
            patient_ids = y_v.index,
            roi = ROI,
            n_workers = n_workers
        )

        return x_tr_gen, x_v_gen, y_tr, y_v