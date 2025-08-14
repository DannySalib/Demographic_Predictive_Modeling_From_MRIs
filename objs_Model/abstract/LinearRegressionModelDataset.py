from objs_Model import ModelDataset
from Types import ROI

class LinearRegressionModelDataset(ModelDataset):
    def __init__(self, participants_df, roi = ROI.PFC):
        super().__init__(participants_df, roi)


    #override
    def get_data(self):
        data = super().get_data()
        # TODO PCA the data
        return data

