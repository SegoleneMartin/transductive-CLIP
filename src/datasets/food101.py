import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxfordpets import OxfordPets


template = 'a photo of {}, a type of food.'


class Food101(DatasetBase):

    def __init__(self, root):
        self.image_dir = os.path.join(root, 'images')
        self.split_path = os.path.join(root, 'split_zhou_Food101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(
            self.split_path, self.image_dir)

        super().__init__(train_x=train, val=val, test=test)
