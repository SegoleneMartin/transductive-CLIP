import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

from .oxfordpets import OxfordPets


template = 'a photo of a person doing {}.'


class UCF101(DatasetBase):

    def __init__(self, root):
        self.image_dir = os.path.join(root, 'UCF-101-midframes')
        self.split_path = os.path.join(root, 'split_zhou_UCF101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(
            self.split_path, self.image_dir)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')[0]  # trainlist: filename, label
                action, filename = line.split('/')
                label = cname2lab[action]

                elements = re.findall('[A-Z][^A-Z]*', action)
                renamed_action = '_'.join(elements)

                filename = filename.replace('.avi', '.jpg')
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=renamed_action
                )
                items.append(item)

        return items
