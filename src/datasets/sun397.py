import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

from .oxfordpets import OxfordPets


template = 'a photo of a {}.'


class SUN397(DatasetBase):

    dataset_dir = 'sun397'

    def __init__(self, root):
        self.image_dir = os.path.join(root, 'SUN397')
        self.split_path = os.path.join(root, 'split_zhou_SUN397.json')

        self.template = template

        train, val, test = OxfordPets.read_split(
            self.split_path, self.image_dir)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(root, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split('/')[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = ' '.join(names)

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)

        return items
