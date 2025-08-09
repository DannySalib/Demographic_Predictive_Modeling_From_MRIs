from itertools import chain
import numpy as np
from nibabel import Nifti1Image
from nilearn import datasets
from nilearn.image import resample_to_img, index_img
from Util.ROIType import ROIType
from nibabel import Nifti1Image

class Atlas:

    def __init__(self):
        self.__data = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
        self.__roi_to_idxs_dict: dict[str, list[int]] = None
    
    @property
    def data(self):
        return self.__data
    
    @property
    def roi_to_idxs_dict(self):
        if self.__roi_to_idxs_dict is None:
            self.__make_roi_dict()

        return self.__roi_to_idxs_dict
    
    @property
    def img(self) -> Nifti1Image:
        return self.__data.maps
    
    def get_roi_mask_fdata(self, roi: ROIType) -> np.ndarray:
        roi_indicies = self.roi_to_idxs_dict.get(roi)

        if roi_indicies is None:
            raise KeyError(f'Unrecognized region of interest: {roi}')
        
        atlas_img_data = self.img.get_fdata()

        # Set a mask of all 0s with the same 3D shape
        mask_fdata = np.zeros(
            shape = self.img.shape[:3],
            dtype=np.uint8
        )

        # For each brain region, use a logical 'or' opperator to update the mask
        # update the mask at coord (x,y,z) to true if the current ROI at coord(x, y, z) > 50% probability 
        # The current ROI at coord (x,y,z) i.e the voxel has a value of 0–100.
        # the value indicates the likelihood (%) that the voxel belongs to a given ROI.
        for roi_idx in roi_indicies:
            roi_prob = atlas_img_data[..., roi_idx]
            mask_fdata |= (roi_prob >= 50)
        
        return mask_fdata
    
    def get_roi_mask(self, roi: str) -> Nifti1Image:
        mask_fdata = self.get_roi_mask_fdata(roi)
        # Return as NIfTI image with same affine and header as the atlas
        return Nifti1Image(
            mask_fdata.astype(np.uint8), 
            affine=self.img.affine, 
            header=self.img.header
        )
    
    # TODO instead of making a whole dict make an init and dynamically create more key/value pairs 
    # when we dont have ROI idxs but we know we can make/find one 
    def __make_roi_dict(self) -> None:
        labels = self.__data.labels[1:] # skip i=0 (background)
        # init roi -> index with labels 
        roi_to_idxs_dict = {label : [i] for i, label in enumerate(labels)}

        # define other brain regions 
        roi_to_idxs_dict[ROIType.PFC.value] = list(chain(
            roi_to_idxs_dict['Frontal Pole'],
            roi_to_idxs_dict['Superior Frontal Gyrus'],
            roi_to_idxs_dict['Middle Frontal Gyrus'],
            roi_to_idxs_dict['Inferior Frontal Gyrus, pars triangularis'],
            roi_to_idxs_dict['Inferior Frontal Gyrus, pars opercularis'],
            roi_to_idxs_dict['Frontal Medial Cortex'],
            roi_to_idxs_dict['Paracingulate Gyrus'],
            roi_to_idxs_dict['Cingulate Gyrus, anterior division'],
            roi_to_idxs_dict['Frontal Orbital Cortex']
        ))

        roi_to_idxs_dict[ROIType.TEMPORAL_LOBE.value] = list(chain(
            roi_to_idxs_dict['Temporal Pole'],
            roi_to_idxs_dict['Superior Temporal Gyrus, anterior division'],
            roi_to_idxs_dict['Superior Temporal Gyrus, posterior division'],
            roi_to_idxs_dict['Middle Temporal Gyrus, anterior division'],
            roi_to_idxs_dict['Middle Temporal Gyrus, posterior division'],
            roi_to_idxs_dict['Middle Temporal Gyrus, temporooccipital part'],
            roi_to_idxs_dict['Inferior Temporal Gyrus, anterior division'],
            roi_to_idxs_dict['Inferior Temporal Gyrus, posterior division'],
            roi_to_idxs_dict['Inferior Temporal Gyrus, temporooccipital part'],
            roi_to_idxs_dict['Parahippocampal Gyrus, anterior division'],
            roi_to_idxs_dict['Parahippocampal Gyrus, posterior division'],
            roi_to_idxs_dict['Temporal Fusiform Cortex, anterior division'],
            roi_to_idxs_dict['Temporal Fusiform Cortex, posterior division'],
            roi_to_idxs_dict['Temporal Occipital Fusiform Cortex'],
            roi_to_idxs_dict['Planum Polare'],
            roi_to_idxs_dict["Heschl's Gyrus (includes H1 and H2)"],
            roi_to_idxs_dict['Planum Temporale']
        ))

        self.__roi_to_idxs_dict = roi_to_idxs_dict