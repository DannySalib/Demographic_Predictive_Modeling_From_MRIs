import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import nibabel as nib

from Types import ROI, Dimension
from objs_MRI import MRIFunc
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

load_roi_expected_errors = (FileNotFoundError, FileExistsError, ValueError, TypeError, KeyError)
def load_roi_fdata(path: str, roi: ROI) -> tuple[np.ndarray | None, Exception | None]:
    '''try to load fMRI as MRIFunc from given path'''
    fMRI: MRIFunc = None
    error: Exception = None
    fdata: np.ndarray = None

    try:
        fMRI = nifti_to_MRIFunc(nib.load(path))
    except load_roi_expected_errors as e:
        error = e

    if fMRI is None:
        return fdata, error

    try:
        fdata = fMRI.get_roi_fdata(roi)
    except load_roi_expected_errors as e:
        error = e

    return fdata, error

class ModelDataset(ABC):
    '''manages the model dataset'''
    def __init__(self, participants_df: pd.DataFrame, roi: ROI = ROI.PFC):
        self._participants_df = shuffle(participants_df)
        calculate_split_idx(participants_df)
        self._roi = roi
        self._X: np.ndarray = None
        self._y: np.ndarray = None

    ### lazy loading ###
    @property
    def X(self) -> np.ndarray:
        '''model data'''
        if self._X is None:
            self._X = self.get_data()

        return self.X

    def get_data(self) -> np.ndarray:
        data = []
        for pid in self._participants_df.participant_id:
            fdata, error = load_roi_fdata(rf'..\..\Data\func\{pid}.nii.gz', self.roi)

            if fdata is None:
                logger.info(f'Could not load pid {pid} roi fdata: {error}')
                continue

            data.append(fdata.flatten())

        # our data should be an array of flattened fMRI fdata
        data = np.stack(data)
        dimension_value = len(data.shape)

        try:
            dimension = Dimension(dimension_value)
        except ValueError as e:
            raise ValueError(f'Unrecognized data dimension: {dimension_value}') from e

        if dimension is not Dimension.TWO_D:
            raise ValueError(f'Unexpected dataset dimensionality: {dimension.value}')

        # drop features with no brain activity
        data = data[:, np.any(data != 0, axis=0)]
        return data

    @property
    def y(self) -> np.array:
        '''data predictions'''
        if self._y is None:
            self._y = self.get_prediction()

        return self._y

    @abstractmethod
    def get_prediction(self) -> np.array:
        pass

    #### non-lazy loading ###
    @property
    def roi(self) -> ROI:
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


