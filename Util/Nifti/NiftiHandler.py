import nibabel as nib 
from nilearn import datasets
import matplotlib.pyplot as plt
import sys 
import os 
# Add the project root to sys.path
sys.path.append(os.path.abspath('../'))
from Util.Nifti.AtlasHandler import AtlasHandler
import numpy as np

class NiftiHandler:
    def __init__(self, path: str):
        self._img = self.update_current_img(path)
        self.__atlas_handler = AtlasHandler()
        
    @property
    def img(self):
        return self._img
    
    @property
    def atlas_handler(self):
        return self.__atlas_handler
    
    @img.setter
    def img(self, path: str):
        self.update_current_img(path)

    # TODO mask the pfc_mask onto the current img 
    # TODO test and see results 
    # TODO explore img smoothing methods, or what other further processing is needed before creating a dataset for a model 
    # TODO generalize methods to process data for all parts of the brain not just PFC
    def get_pfc_img_data(self) -> np.array:
        if not self.is_normalized():
            self.normalize()
        
        return 
    
    def is_normalized(self):
        atlas_img = self.__atlas_handler.get_img()
        return np.allclose(atlas_img.affine, atlas_img.affine, atol=1e-2)

    def normalize(self):
        return 

    def update_current_img(self, path: str):
        try:
            self._img = nib.load(path)
        except Exception as e:
            raise Exception(f'Error loading nifti file: {e}')

    def plot_img(self, nrows: int = 5, ncols: int = 5):
        fig, axis = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            figsize = (10, 10)
        )

        img_data = self._img.get_fdata()
        zMax = img_data.shape[2]
        step = zMax // (nrows*ncols) # n^2 = total number of slices we're graphing 

        current_slice = 0 # index of current slice from the MRI scan
        # for each axis ...
        for i in range(nrows):
            for j in range(ncols):

                axis[i][j].imshow(
                    img_data[:, :, current_slice],
                    cmap = 'gray',
                )

                axis[i][j].axis('off')

                current_slice += step
        
        return fig
    

