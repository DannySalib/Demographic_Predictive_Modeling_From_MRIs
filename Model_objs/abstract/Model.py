from abc import ABC, abstractmethod
import os
import warnings
import numpy as np
import pandas as pd


import nibabel as nib
from multiprocessing import Pool, cpu_count
from typing import Generator
# from Atlas import Atlas
from Model_objs.abstract import ModelDataset

PFC_ROI_STR: str = 'Prefrontal Cortex'
MIN_PROCESSOR_CORES: int = 4

class Model(ABC):
    '''abstract model for all the different ML models i decide to make or use'''
    def __init__(self, participants_df: pd.DataFrame):
        self._particpants_df = participants_df
        self._dataset: ModelDataset = None

    @property
    def dataset(self) -> ModelDataset:
        '''dataset of model'''
        return self.get_dataset()

    def get_dataset(self) -> ModelDataset:
        '''get the data set of the model'''
        if self._dataset is None:
            self._dataset = self._create_dataset()

        return self._dataset


    @abstractmethod
    def run(self) -> None:
        '''runs model'''
        pass

    @abstractmethod
    def _create_dataset(self) -> ModelDataset:
        '''loads the Model's dataset'''
        pass
