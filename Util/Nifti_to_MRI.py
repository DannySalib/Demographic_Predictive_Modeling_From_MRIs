
from nibabel import Nifti1Image
from Util.MRI4D import MRI4D

def nifti_to_MRI4D(img: Nifti1Image) -> MRI4D:
    return MRI4D(img.dataobj, img.affine, img.header)

