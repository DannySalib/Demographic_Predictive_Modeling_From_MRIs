
from Util.MRI4D import MRI4D
from nibabel import Nifti1Image

class MRIFunc(MRI4D):
    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None, dtype=None):
        super().__init__(dataobj, affine, header, extra, file_map, dtype)
        