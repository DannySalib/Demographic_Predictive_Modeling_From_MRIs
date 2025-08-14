from objs_MRI.abstract import MRI4D

class MRIFunc(MRI4D):
    '''fMRI nifti handler'''
    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None, dtype=None, **kwargs):
        super().__init__(dataobj, affine, header, extra, file_map, dtype)
