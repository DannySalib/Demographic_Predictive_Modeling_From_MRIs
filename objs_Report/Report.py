import os
import pandas as pd
from objs_Model import BrainAgePredictor, Model
from Types import Predictor

try:
    cwd = os.getcwd()
    participants_data_path = f'{cwd}../Data/participants.tsv'

    df_participants = pd.read_csv(participants_data_path, sep='\t')
except (OSError, FileNotFoundError, PermissionError, IOError) as e:
    raise OSError('Error while retrieving participants df') from e

class Report:
    """
    Run model and generate a report
    """
    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def get_model(self) -> Model:
        """Based on the prediction type, return the corresponding model"""
        match self._predictor:
            case Predictor.BRAIN_AGE:
                return BrainAgePredictor(df_participants)
            case _:
                raise ValueError(f'Unrecognized predictor: {self._predictor.name}')

    def run(self):
        """run report"""
        model = self.get_model()
        model.run()
        return model