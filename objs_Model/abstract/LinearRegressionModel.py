from abc import ABC
from objs_Model import Model

class LinearRegressionModel(Model):

    def __init__(self, participants_df):
        super().__init__(participants_df)

