from os.path import getmtime as os_path_getmtime
from pathlib import Path
from copy import deepcopy


def slicing(listy, indices):
    return [listy[i] for i in indices]


# If I have an index mapping, then that's all I need.
class RelPainter:
    def __init__(self, dirpath_aug):
        self.dirpath_aug = Path(dirpath_aug)

    def convert(self, vgdataset):
        '''
        _summary_

        Args:
            vgdataset (VGDataset): _description_

        Raises:
            AssertionError: _description_

        Returns:
            _type_: _description_
        '''
        vgdataset = deepcopy(vgdataset)
        # vgdataset.filenames = self.map_filenames(vgdataset.filenames)
        # self.filenames_new, self.indices_new = self.map_filenames(vgdataset.filenames)
        filenames_new, indices_new = self.map_filenames(vgdataset.filenames)
        print(filenames_new[0])
        filenames_new_flat = sum(filenames_new, [])
        vgdataset.filenames = filenames_new_flat
        indices_new_flat = sum(indices_new, [])
        vgdataset.img_info = slicing(vgdataset.img_info, indices_new_flat)
        vgdataset.gt_boxes = slicing(vgdataset.gt_boxes, indices_new_flat)
        vgdataset.gt_classes = slicing(vgdataset.gt_classes, indices_new_flat)
        vgdataset.gt_attributes = slicing(
            vgdataset.gt_attributes, indices_new_flat)
        vgdataset.relationships = slicing(
            vgdataset.relationships, indices_new_flat)
        return vgdataset

    def map_filenames(self, filenames: list[str]) -> list[list[str]]:
        list_filenames = []
        list_indices = []
        dirpath_aug = self.dirpath_aug
        for idx, filename in enumerate(filenames):
            dirpath_augs_per_image = dirpath_aug / Path(filename).stem
            if not dirpath_augs_per_image.is_dir():
                continue
            filenames_augs_per_image = sorted(
                list(dirpath_augs_per_image.glob('*')), key=os_path_getmtime)
            list_filenames.append(filenames_augs_per_image)
            list_indices.append([idx] * len(filenames_augs_per_image))

        return list_filenames, list_indices
