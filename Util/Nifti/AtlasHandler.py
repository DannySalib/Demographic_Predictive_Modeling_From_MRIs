import nibabel as nib #type: ignore
from nilearn import datasets
import numpy as np

class AtlasHandler:
    def __init__(self):
        self.__atlas  = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')

        self.__atlas_label_to_index = {label : i for i, label in enumerate(self.__atlas.labels[1:])} # skip i=0 (background)

    @property
    def atlas(self):
        return self.__atlas
    
    @property
    def atlas_label_to_index(self):
        return self.__atlas_label_to_index
    
    def get_img(self) -> nib.Nifti1Image:
        return self.__atlas.maps
    
    def get_pfc_roi_indicies(self) -> list:
        return [
            self.__atlas_label_to_index['Frontal Pole'],
            self.__atlas_label_to_index['Superior Frontal Gyrus'],
            self.__atlas_label_to_index['Middle Frontal Gyrus'],
            self.__atlas_label_to_index['Inferior Frontal Gyrus, pars triangularis'],
            self.__atlas_label_to_index['Inferior Frontal Gyrus, pars opercularis'],
            self.__atlas_label_to_index['Frontal Medial Cortex'],
            self.__atlas_label_to_index['Paracingulate Gyrus'],
            self.__atlas_label_to_index['Cingulate Gyrus, anterior division'],
            self.__atlas_label_to_index['Frontal Orbital Cortex']
        ]
    
    def get_pfc_mask(self) -> np.array:
        atlas_img = self.__atlas.maps
        atlas_img_data = atlas_img.get_fdata()

        # Set a mask of all 0s with the same 3D shape
        pfc_mask = np.zeros(
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
            pfc_mask |= (roi_prob >= 50)

        return pfc_mask

    

