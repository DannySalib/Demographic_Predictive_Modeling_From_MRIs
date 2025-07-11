import nibabel as nib
from nibabel import Nifti1Image
from nilearn.image import resample_to_img, index_img
import matplotlib.pyplot as plt
import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath('../'))
from Util.Nifti.AtlasHandler import AtlasHandler
import numpy as np
from warnings import warn

# TODO major refactor needed:
#   - The TASK data is the data we need to train on. this is
#       4D data. How do we handle them accordingly?
class NiftiHandler:
    def __init__(self, img: Nifti1Image = None):
        self._atlas_handler = AtlasHandler()

        self._img: Nifti1Image = None
        self._img_dim: int = None
        if img: # this defines self.img
            self.update(img)

    def update(self, img: Nifti1Image) -> None:
        self._img = img
        self._img_dim = len(self._img.shape)

        if not self.is_3D and not self.is_4D:
            raise Exception(f'Cannot work with {self._img_dim}D data')

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img: Nifti1Image):
        self.update(img)

    @property
    def atlas_handler(self):
        return self._atlas_handler

    @property
    def img_dim(self):
        return self.img_dim

    @property
    def is_3D(self):
        return self._img_dim == 3

    @property
    def is_4D(self):
        return self._img_dim == 4

    def correct_for_motion(self) -> None:
        if not self.is_4D:
            warn('This function is only needed for 4D fMRI data. Your dimension: {self._img_dim}\nreturning...')
            return

        motion_corrected = []
        # Align all other MRIs @ t = t to MRI @ t = 0
        refrence_img = index_img(self._img, 0)
        for t in range(self._img.shape[-1]):
            curr_img = index_img(self._img, t)
            corrected = resample_to_img(curr_img, refrence_img)
            motion_corrected.append(corrected.get_fdata())

        motion_corrected = np.stack(motion_corrected, axis=-1)
        self._img = Nifti1Image(motion_corrected, self._img.affine, self._img.header)

    def get_roi_img(self, roi: str) -> Nifti1Image:
        if not self.is_affine_normalized():
            self.normalize_by_affine()

        mask = self.atlas_handler.get_roi_mask(roi)

        # error happens when applying a 2mm resolution mask to a 1mm resolution image
        if self._img.shape != mask.shape:
            mask = self.resample_mask_shape(mask)

        # float32 since apparently were still tight on memory in this day and age
        img_data = self._img.get_fdata(dtype=np.float32)
        mask_data = mask.get_fdata(dtype=np.float32)
        mask_data = mask_data.astype(np.bool_)

        # Earlier we asserted that the dimension must either be 3D or 4D
        if self.is_3D:
            # Lets superficially add a new dimension such that
            # the masking below can be generalized for both 3D/4D
            img_data = img_data[..., np.newaxis]

        masked_img_data = np.zeros_like(img_data, np.float64)
        for t in range(img_data.shape[-1]):
            # Apply the binary PFC mask
            masked_img_data[..., t] = img_data[..., t] * mask_data

        # Change 3D from 4D back to 3D
        if self.is_3D:
            masked_img_data = masked_img_data[..., 0]

        return Nifti1Image(
            masked_img_data,
            affine = self._img.affine,
            header = self._img.header
        )

    def resample_mask_shape(self, mask: Nifti1Image) -> Nifti1Image:
        if self.is_3D:
            refrence = self._img
        else: # must be 4D
            refrence = index_img(self.img, 0)

        return resample_to_img(
            mask,
            refrence,
            interpolation='nearest'
        )

    def is_affine_normalized(self) -> bool:
        atlas_img = self._atlas_handler.get_img()
        return np.allclose(atlas_img.affine, atlas_img.affine, atol=1e-2)

    def normalize_by_affine(self) -> None:
        #mni_data = resample_to_img(your_mri_img, atlas_img, interpolation='continuous')
        # something like that
        raise NotImplementedError()

    def show_img(self, nrows: int = 5, ncols: int = 5, time: int = None) -> None:
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

        img_data = self._img.get_fdata()

        # I want to be able to graph atlas `Nifti1Image`s
        time_defined = time is not None
        if self.is_3D and time_defined:
            warn('show_img: Cannot use ROI_index on 3D img...')

        if self.is_4D:
            if time_defined:
                img_data = img_data[..., time]
            else:
                raise Exception(f'Cannot graph 4D img with undefined ROI_index')

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
        return str(self._img)


