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

    def update_current_img(self, path: str):
        try:
            self._img = nib.load(path)
        except Exception as e:
            raise Exception(f'Error loading nifti file: {e}')

    def get_pfc_img_data(self):
        atlas_img = self.__atlas_handler.atlas.maps
        atlas_img_data = atlas_img.get_fdata()

        # Set a mask of all 0s with the same 3D shape
        pfc_img_data = np.zeros(
            shape = atlas_img.shape[:3],
            dtype=bool
        )

        pfc_roi_indicies = self.__atlas_handler.get_pfc_roi_indicies()
        # For each brain region, use a logical or to update the mask
        # update the mask at coord (x,y,z) to true if the current ROI at coord(x, y, z) > 50% probability 
        # The current ROI at coord (x,y,z) i.e the voxel has a value of 0–100.
        # the value indicates the likelihood (%) that the voxel belongs to a given ROI.
        for roi_idx in pfc_roi_indicies:
            roi_prob = atlas_img_data[..., roi_idx]
            pfc_img_data |= (roi_prob >= 50)

        return pfc_img_data

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
    

