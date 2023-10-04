import os
import random
from scipy.io import loadmat
from collections import defaultdict

from .oxfordpets import OxfordPets
from .utils import Datum, DatasetBase, read_json


template = 'a photo of a {}, a type of flower.'


class Flowers102(DatasetBase):


    def __init__(self, root):
        self.image_dir = os.path.join(root, 'jpg')
        self.label_file = os.path.join(root, 'imagelabels.mat')
        self.lab2cname_file = os.path.join(root, 'cat_to_name.json')
        self.split_path = os.path.join(root, 'split_zhou_OxfordFlowers.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        #train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)['labels'][0]
        for i, label in enumerate(label_file):
            imname = f'image_{str(i + 1).zfill(5)}.jpg'
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)
        
        print('Splitting data into 50% train, 20% val, and 30% test')

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=y-1, # convert to 0-based label
                    classname=c
                )
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train:n_train+n_val], label, cname))
            test.extend(_collate(impaths[n_train+n_val:], label, cname))
        
        return train, val, test