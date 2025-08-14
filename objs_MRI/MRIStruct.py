
from nibabel import Nifti1Image
from objs_MRI import MRI3D

class MRIStruct(MRI3D):
    def __init__(self, img: Nifti1Image):
        super().__init__(img)