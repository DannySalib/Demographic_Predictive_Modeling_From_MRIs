
from Util.MRI3D import MRI3D
from nibabel import Nifti1Image

class MRIStruct(MRI3D):
    def __init__(self, img: Nifti1Image):
        super().__init__(img)