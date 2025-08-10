import numpy as np

from Model_objs.abstract import LinearRegressionModel, LinearRegressionModelDataset
from enums import ROIType

class BrainAgePredictorDataset(LinearRegressionModelDataset):
    def __init__(self, participants_df, roi=ROIType.PFC):
        super().__init__(participants_df, roi)

    ########## implement abstract methods
    def _create_prediction(self) -> np.array:
        return self._participants_df.Age.values

class BrainAgePredictor(LinearRegressionModel):

    def __init__(self, participants_df):
        super().__init__(participants_df)

    ########## implement abstract methods
    def run(self) -> None:
        '''runs model'''
        return


    def _create_dataset(self) -> BrainAgePredictorDataset:
        '''loads the Model's dataset'''
        return BrainAgePredictorDataset(self._particpants_df)
