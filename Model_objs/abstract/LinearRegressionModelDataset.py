from Model_objs.abstract import ModelDataset
from enums import ROIType

class LinearRegressionModelDataset(ModelDataset):
    def __init__(self, participants_df, roi = ROIType.PFC):
        super().__init__(participants_df, roi)


    #override
    def _create_data(self):
        data = super()._create_data()
        # TODO PCA the data
        return data

