'''
Represent MRI data as a Nifti1Image obj
'''

from abc import ABC, abstractmethod
import numpy as np
from nibabel import Nifti1Image
from nilearn.image import resample_to_img, index_img
import matplotlib.pyplot as plt
from Types import ROI, Dimension
from objs_MRI import Atlas

class MRI(Nifti1Image, ABC):
    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None, dtype=None):
        super().__init__(dataobj, affine, header, extra, file_map, dtype)
        self.__atlas = Atlas()
        self._current_roi_mask_fdata: np.ndarray = None
        self._current_roi: ROI = None

    ####### Abstract functionality
    @property
    @abstractmethod
    def dimension(self) -> Dimension:
        '''dimensionality of img'''
        pass

    @abstractmethod
    def resample(self, refrence: 'MRI3D') -> 'MRI4D':
        '''resamples the fdata cache'''
        pass

    @abstractmethod
    def correct_for_motion(self) -> None:
        pass

    ###### Shared functionality
    @property
    def current_roi(self) -> str:
        return self._current_roi

    @property
    def current_roi_mask_fdata(self) -> np.ndarray:
        return self._current_roi_mask_fdata

    def get_roi_mask(self, roi: str) -> Nifti1Image:
        return self.__atlas.get_roi_mask(roi)

    def resample_mask(self, roi_mask: 'MRIMask') -> Nifti1Image:
        return resample_to_img(
            source_img=roi_mask,
            target_img=index_img(self, index=0),
            interpolation='nearest'
        )

    def get_roi_mask_fdata(self, roi: ROI) -> np.ndarray:
        roi_mask_img = self.get_roi_mask(roi)
        # error happens when applying a 2mm resolution mask to a 1mm resolution image
        if self.shape != roi_mask_img.shape:
            roi_mask_img = self.resample_mask(roi_mask_img)

        # load as float 32 first before converting to bool
        # avoids memory errors
        roi_mask_fdata = roi_mask_img.get_fdata(dtype=np.float32)
        roi_mask_fdata = roi_mask_fdata.astype(np.bool_)

        return roi_mask_fdata

    def get_roi_fdata(self, roi: ROI) -> np.ndarray:
        if self.current_roi != roi:
            self._current_roi_mask_fdata = self.get_roi_mask_fdata(roi)
            self._current_roi = roi

        # float32 since apparently were still tight on memory in this day and age
        img_data = self.get_fdata(dtype=np.float32)

        if self.dimension is Dimension.THREE_D:
            return img_data * self.current_roi_mask_fdata

        img_data_dimension = len(img_data.shape)
        assert img_data_dimension == Dimension.FOUR_D.value

        # determine the shape of the img_data after applying an roi mask
        nt = img_data.shape[-1]
        t0 = 0
        masked_img_data_init = img_data[..., t0] * self.current_roi_mask_fdata
        masked_img_data_shape = masked_img_data_init.shape + (nt,)

        # use that to initialize an NxD array
        masked_img_data = np.zeros(masked_img_data_shape, np.float64)
        masked_img_data[..., t0] = masked_img_data_init

        for t in range(1, nt):
            # Apply the binary PFC mask
            masked_img_data[..., t] = img_data[..., t] * self.current_roi_mask_fdata

        return masked_img_data

    def get_roi_img(self, roi: ROI) -> Nifti1Image:
        """Returns the region of interest of the current img

        Args:
            roi (str): _description_

        Returns:
            _type_: _description_
        """

        img_data = self.get_roi_fdata(roi, preserve_shape=True)

        return Nifti1Image(
            img_data,
            affine=self.affine,
            header=self.header
        )


    def show(self, nrows: int = 5, ncols: int = 5, at_time: int = 0) -> None:
        '''show a plot of the current img'''
        n_limit = 2
        if nrows < n_limit:
            raise ValueError(f'nrows must be greater than {n_limit}')
        if ncols < n_limit:
            raise ValueError(f'ncols must be greater than {n_limit}')

        fig, axis = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            figsize = (10, 10)
        )

        img_data = self.get_fdata()
        if self.dimension is Dimension.THREE_D:
            img_data = img_data[..., np.newaxis]

        img_data_dimension = len(img_data.shape)
        if img_data_dimension != Dimension.FOUR_D.value:
            raise RuntimeError(f'Cannot show img where {img_data_dimension = }')

        num_slices = nrows * ncols

        nz = img_data.shape[2]
        zstep = nz // num_slices

        nt = img_data.shape[3]
        if at_time > (nt-1):
            raise ValueError(f'Cannot show MRI {at_time = } (out of bounds)')

        current_slice = 0 # index of current slice from the MRI scan
        # for each axis ...
        for i in range(nrows):
            for j in range(ncols):

                axis[i][j].imshow(
                    img_data[:, :, current_slice, at_time],
                    cmap = 'gray',
                )

                axis[i][j].axis('off')

                current_slice += zstep







