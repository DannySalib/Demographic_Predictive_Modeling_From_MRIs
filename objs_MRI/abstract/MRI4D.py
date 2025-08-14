
import numpy as np
from nilearn.image import resample_to_img, index_img
from objs_MRI import MRI
from Types import Dimension

class MRI4D(MRI):
    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None, dtype=None):
        super().__init__(dataobj, affine, header, extra, file_map, dtype)

#        if len(self.shape) != Dimension.FOUR_D.value:
#            raise ValueError(f'Cannot inititalize MRI 4D non 4D data shape: {self.shape}')

    ######## Mandatory abstract methods
    @property
    def dimension(self) -> Dimension:
        return Dimension.FOUR_D

    def resample(self, refrence: 'MRI3D') -> 'MRI4D':
        """Updates fdata cache with resampled fdata

        Args:
            refrence (MRI): img to resample to
            interpolation (str, optional): Interpolaiton type. Defaults to 'continuous'.
        """
        refrence_dimension = len(refrence.shape)
        assert refrence_dimension == Dimension.THREE_D.value

        resampled_shape = refrence.shape + (self.nt,)
        resampled = np.zeros(resampled_shape, dtype=self.get_data_dtype())
        # Align all other MRIs @ t = t to MRI @ t = 0
        for t in range(self.nt):
            curr_img = index_img(self, t)
            curr_img_corrected = resample_to_img(curr_img, refrence)

            resampled[..., t] = curr_img_corrected.get_fdata()

        return MRI4D(
            resampled,
            affine=self.affine,
            header=self.header
        )

    def correct_for_motion(self) -> None:
        """Functional MRI data needs to account for head motion

        Returns:
            Self: 4D MRI
        """
        self.resample(
            refrence = index_img(self, index=0)
        )

    ##### Unique methods to a 4D MRI
    @property
    def nt(self) -> int:
        return self.shape[-1]

    def get_tsnr(self):
        '''temporal signal-to-noise ratio'''
        data = self.get_fdata()

        tsnr: float = None
        try:
            tsnr = np.mean(data, -1) / np.std(data, -1)
        except ZeroDivisionError:
            # todo log?
            pass # return as none

        return tsnr
