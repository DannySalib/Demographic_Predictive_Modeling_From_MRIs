import nibabel as nib
from nibabel import Nifti1Image
from nilearn import datasets
import numpy as np
from itertools import chain
class AtlasHandler:
    def __init__(self):
        self.__atlas  = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
        self.__roi_to_atlas_index_dict = self.__create_roi_to_atlas_index_dict()

    def __create_roi_to_atlas_index_dict(self) -> dict[str, int]:
        labels = self.__atlas.labels[1:] # skip i=0 (background)
        # init roi -> index with labels 
        roi_to_indices = {label : [i] for i, label in enumerate(labels)}

        # define other brain regions 
        roi_to_indices['Prefrontal Cortex'] = list(chain(
            roi_to_indices['Frontal Pole'],
            roi_to_indices['Superior Frontal Gyrus'],
            roi_to_indices['Middle Frontal Gyrus'],
            roi_to_indices['Inferior Frontal Gyrus, pars triangularis'],
            roi_to_indices['Inferior Frontal Gyrus, pars opercularis'],
            roi_to_indices['Frontal Medial Cortex'],
            roi_to_indices['Paracingulate Gyrus'],
            roi_to_indices['Cingulate Gyrus, anterior division'],
            roi_to_indices['Frontal Orbital Cortex']
        ))

        roi_to_indices['Temporal Lobe'] = list(chain(
            roi_to_indices['Temporal Pole'],
            roi_to_indices['Superior Temporal Gyrus, anterior division'],
            roi_to_indices['Superior Temporal Gyrus, posterior division'],
            roi_to_indices['Middle Temporal Gyrus, anterior division'],
            roi_to_indices['Middle Temporal Gyrus, posterior division'],
            roi_to_indices['Middle Temporal Gyrus, temporooccipital part'],
            roi_to_indices['Inferior Temporal Gyrus, anterior division'],
            roi_to_indices['Inferior Temporal Gyrus, posterior division'],
            roi_to_indices['Inferior Temporal Gyrus, temporooccipital part'],
            roi_to_indices['Parahippocampal Gyrus, anterior division'],
            roi_to_indices['Parahippocampal Gyrus, posterior division'],
            roi_to_indices['Temporal Fusiform Cortex, anterior division'],
            roi_to_indices['Temporal Fusiform Cortex, posterior division'],
            roi_to_indices['Temporal Occipital Fusiform Cortex'],
            roi_to_indices['Planum Polare'],
            roi_to_indices["Heschl's Gyrus (includes H1 and H2)"],
            roi_to_indices['Planum Temporale']
        ))

        return roi_to_indices
        
    @property
    def atlas(self):
        return self.__atlas
    
    @property
    def roi_to_atlas_index_dict(self):
        return self.__roi_to_atlas_index_dict
    
    def get_img(self) -> Nifti1Image:
        return self.__atlas.maps
    
    def get_roi_mask(self, roi: str) -> Nifti1Image:
        roi_indicies = self.__roi_to_atlas_index_dict.get(roi)

        if not roi_indicies:
            raise Exception(f'Unrecognized region of interest: {roi}')
        
        atlas_img = self.__atlas.maps
        atlas_img_data = atlas_img.get_fdata()

        # Set a mask of all 0s with the same 3D shape
        mask_fdata = np.zeros(
            shape = atlas_img.shape[:3],
            dtype=np.uint8
        )

        # For each brain region, use a logical 'or' opperator to update the mask
        # update the mask at coord (x,y,z) to true if the current ROI at coord(x, y, z) > 50% probability 
        # The current ROI at coord (x,y,z) i.e the voxel has a value of 0–100.
        # the value indicates the likelihood (%) that the voxel belongs to a given ROI.
        for roi_idx in roi_indicies:
            roi_prob = atlas_img_data[..., roi_idx]
            mask_fdata |= (roi_prob >= 50)

        # Return as NIfTI image with same affine and header as the atlas
        return nib.Nifti1Image(
            mask_fdata.astype(np.uint8), 
            affine=atlas_img.affine, 
            header=atlas_img.header
        )
        
