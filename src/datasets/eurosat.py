import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxfordpets import OxfordPets


template = 'a centered satellite photo of {}.'


NEW_CLASSNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSAT(DatasetBase):

    def __init__(self, root):
        self.image_dir = os.path.join(root, 'images')
        self.split_path = os.path.join(root, 'split_zhou_EuroSAT.json')

        self.template = template

        train, val, test = OxfordPets.read_split(
            self.split_path, self.image_dir)

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            cname_new = cname_old
            item_new = Datum(
                impath=item_old.impath,
                label=item_old.label,
                classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
