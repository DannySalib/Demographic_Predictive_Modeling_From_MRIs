import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import nibabel as nib

from enums import ROIType
from MRI_objs import MRIFunc
from helper_funcs import nifti_to_MRIFunc

########### Create and configure logger ##############
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info("Just an information")

######### split idx helper ###########
__split_idx: int = None

def calculate_split_idx(df: pd.DataFrame) -> None:
    '''calculates the split index of a df'''
    __split_idx = len(df) * 2 // 3

@lambda _: _()
def split_idx():
    '''get the stored split idx'''
    if __split_idx is None:
        raise ValueError('Calculate split idx first!')
    return __split_idx

####### Helper functions #########
def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    '''shuffle the given df with a set seed'''
    return df.sample(frac=1, random_state=42)

def load_MRIFunc_roi_fdata(path: str, roi: ROIType) -> tuple[np.ndarray | None, Exception | None]:
    '''try to load fMRI as MRIFunc from given path'''
    fMRI: MRIFunc = None
    error: Exception = None
    fdata: np.ndarray = None

    expected_errors = (FileNotFoundError, FileExistsError, ValueError, TypeError, KeyError)

    try:
        fMRI = nifti_to_MRIFunc(nib.load(path))
    except expected_errors as e:
        error = e

    if fMRI is None:
        return fdata, error

    try:
        fdata = fMRI.get_roi_fdata(roi)
    except expected_errors as e:
        error = e

    return fdata, error

class ModelDataset(ABC):
    '''manages the model dataset'''
    def __init__(self, participants_df: pd.DataFrame, roi: ROIType = ROIType.PFC):
        self._participants_df = shuffle(participants_df)
        calculate_split_idx(participants_df)

        self._roi = roi
        self._X: np.ndarray = None
        self._y: np.ndarray = None

    ### lazy loading ###
    @property
    def X(self) -> np.ndarray:
        '''model data'''
        return self.get_data()

    def get_data(self) -> np.ndarray:
        '''get model data'''
        if self._X is None:
            self._X = self._create_data()

        return self._X

    def _create_data(self) -> np.ndarray:
        data = []
        for pid in self._participants_df.participant_id:
            fdata, error = load_MRIFunc_roi_fdata(rf'..\..\Data\func\{pid}.nii.gz', self.roi)

            if fdata is None:
                logger.info(f'Could not load pid {pid} roi fdata: {error}')
                continue

            data.append(fdata.flatten())

        # our data should be an array of flattened fMRI fdata
        data = np.stack(data)
        assert len(data.shape) == 2, f'Cannot use model data with unexpected dimension: {len(data.shape) != 2}'

        # drop features with no brain activity
        data = data[:, np.any(data != 0, axis=0)]
        return data

    @property
    def y(self) -> np.array:
        '''data prediction'''
        return self.get_prediction()

    def get_prediction(self) -> np.array:
        '''gets data prediction'''
        if self._y is None:
            self._y = self._create_prediction()

        return self._y

    @abstractmethod
    def _create_prediction(self) -> np.array:
        pass

    #### non-lazy loading ###
    @property
    def roi(self) -> ROIType:
        '''read only'''
        return self._roi

    @property
    def Xtr(self) -> np.ndarray:
        '''Training data of model dataset'''
        return self.get_data_training()

    def get_data_training(self) -> np.ndarray:
        '''Get training data of model dataset'''
        return self.X[:split_idx, :]

    @property
    def Xv(self) -> np.ndarray:
        '''Validating data of model dataset'''
        return self.get_data_validating()

    def get_data_validating(self) -> np.ndarray:
        '''get the validating data of model dataset'''
        return self.X[split_idx:, :]

    @property
    def ytr(self) -> np.ndarray:
        return self.get_prediction_training()

    def get_prediction_training(self) -> np.ndarray:
        '''Get the prediction of model training data'''
        return self.y[:split_idx, :]

    @property
    def yv(self) -> np.ndarray:
        return self.get_prediction_validating()

    def get_prediction_validating(self) -> np.ndarray:
        return self.y[split_idx:, :]


