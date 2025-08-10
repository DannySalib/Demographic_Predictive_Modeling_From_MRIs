
from nibabel import Nifti1Image
from MRI_objs import MRIFunc

def nifti_to_MRIFunc(img: Nifti1Image) -> MRIFunc:
    return MRIFunc(img.dataobj, img.affine, img.header)
