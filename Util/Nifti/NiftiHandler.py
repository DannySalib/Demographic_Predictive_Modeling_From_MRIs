import nibabel as nib 
from nibabel import Nifti1Image
from nilearn import datasets
import matplotlib.pyplot as plt
import sys 
import os 
# Add the project root to sys.path
sys.path.append(os.path.abspath('../'))
from Util.Nifti.AtlasHandler import AtlasHandler
import numpy as np
from warnings import warn
from nilearn.image import resample_to_img
from nptyping import NDArray

class NiftiHandler:
    def __init__(self, path: str = None):
        self.img: Nifti1Image = None

        if path:
            self.update_current_img_with_path(path)

        self._atlas_handler = AtlasHandler()
    
    @property
    def atlas_handler(self):
        return self._atlas_handler
    
    # TODO explore img smoothing methods, or what other further processing is needed before creating a dataset for a model 
    # TODO generalize methods to process data for all parts of the brain not just PFC
    def get_roi_img(self, roi: str) -> Nifti1Image:
        if not self.is_affine_normalized():
            self.normalize_by_affine()
        
        mask = self.atlas_handler.get_roi_mask(roi)

        # error happens when applying a 2mm resolution mask to a 1mm resolution image
        if self.img.shape != mask.shape:
            mask = self.resample_mask_shape(mask)
        
        # float32 since apparently were still tight on memory in this day and age
        img_data = self.img.get_fdata(dtype=np.float32) 
        mask_data = mask.get_fdata(dtype=np.float32)
        mask_data = mask_data.astype(np.bool_)

        # Apply the binary PFC mask
        img_data_masked = img_data * mask_data

        return Nifti1Image(
            img_data_masked,
            affine = self.img.affine,
            header = self.img.header
        )
    
    def resample_mask_shape(self, mask: np.array) -> NDArray:
        return resample_to_img(
            mask, 
            self.img, 
            interpolation='nearest'
        )

    def is_affine_normalized(self) -> bool:
        atlas_img = self._atlas_handler.get_img()
        return np.allclose(atlas_img.affine, atlas_img.affine, atol=1e-2)

    def normalize_by_affine(self) -> None:
        #mni_data = resample_to_img(your_mri_img, atlas_img, interpolation='continuous')
        # something like that 
        raise NotImplementedError()

    def update_current_img_with_path(self, path: str) -> None:
        try:
            self.img = nib.load(path)
        except Exception as e:
            raise Exception(f'Error loading nifti file: {e}')

    def show_img(self, nrows: int = 5, ncols: int = 5, ROI_index: int = None) -> None:
        n_limit = 2
        if nrows < n_limit:
            raise Exception(f'nrows must be greater than {n_limit}')
        if ncols < n_limit:
            raise Exception(f'ncols must be greater than {n_limit}')
        
        fig, axis = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            figsize = (10, 10)
        )

        img_data = self.img.get_fdata()

        # I want to be able to graph atlas `Nifti1Image`s
        dimension = len(img_data.shape)
        if dimension == 3 and ROI_index is not None:
            warn('show_img: Cannot use ROI_index on 3D img...')
        if dimension == 4:
            if ROI_index:
                img_data = img_data[..., ROI_index]
            else:
                raise Exception(f'Cannot graph 4D img with undefined ROI_index')
        elif dimension > 4:
            raise Exception(f'Cannot graph img with {dimension = }')
        
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
    
    def __str__(self):
        return str(self.img)
    

