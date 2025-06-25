import nibabel as nib #type: ignore
from nilearn import datasets

class AtlasHandler:
    def __init__(self):
        self.__atlas  = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')

        self.__atlas_label_to_index = {label : i for i, label in enumerate(self.__atlas.labels[1:])} # skip i=0 (background)

    @property
    def atlas(self):
        return self.__atlas
    
    @property
    def atlas_label_to_index(self):
        return self.__atlas_label_to_index
    
    def get_pfc_roi_indicies(self) -> list:
        return [
            self.__atlas_label_to_index['Frontal Pole'],
            self.__atlas_label_to_index['Superior Frontal Gyrus'],
            self.__atlas_label_to_index['Middle Frontal Gyrus'],
            self.__atlas_label_to_index['Inferior Frontal Gyrus, pars triangularis'],
            self.__atlas_label_to_index['Inferior Frontal Gyrus, pars opercularis'],
            self.__atlas_label_to_index['Frontal Medial Cortex'],
            self.__atlas_label_to_index['Paracingulate Gyrus'],
            self.__atlas_label_to_index['Cingulate Gyrus, anterior division'],
            self.__atlas_label_to_index['Frontal Orbital Cortex']
        ]
    

