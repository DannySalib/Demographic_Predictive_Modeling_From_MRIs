from dataclasses import dataclass
import pandas as pd 
import numpy as np
from Util.MRIDimension import MRIDimension

class ModelDataSet:
    def __init__(self):
        self.split_idx: int = None
        self.num_entries: int = None
        
        self.X: np.ndarray = None 
        self.y: np.array = None

    @property
    def current_dimension(self):
        if self.X is None:
            return None
        
        return len(self.X.shape)     
    
    @property
    def Xtr(self) -> np.ndarray:
        return self.X[:self.split_idx, :]
    
    @property
    def Xv(self) -> np.ndarray:
        return self.X[self.split_idx:, :]

    @property
    def ytr(self) -> np.ndarray:
        return self.y[:self.split_idx, :]
    
    @property
    def yv(self) -> np.ndarray:
        return self.y[self.split_idx:, :]




class ModelDataHandler:
    def __init__(self, participants_df: pd.DataFrame):
        self._participants_df = participants_df 
        self._model_dataset = ModelDataSet()
        self._Xtr: np.ndarray = None

    @property
    def participant_df(self) -> pd.DataFrame:
        return self._participants_df
    
    @property
    def model_dataset(self) -> ModelDataSet:
        return self._model_dataset
    
    @property
    def Xtr(self) _> 
    
    def get_shuffled_participants_df(self, random_state: int = 42) -> pd.DataFrame:
        """Shuffles dataframe with a random seed"""
        return self.participant_df.sample(
            frac = 1,
            random_state = random_state
        )

    def get_split_idx(self) -> int:
        '''returns the index where current df is split into
        2/3rds training 1/3rd validating'''
        nrows, _ = self.participant_df.shape
        numer, denom = 2, 3 # 2/3rd Training 1/3rd Validating
        return nrows * numer // denom


    