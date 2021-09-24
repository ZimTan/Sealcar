import os
import random
import torch

from torch.utils.data import Dataset

import numpy as np

from json import load as jsonload


class MyDataSet(Dataset):

    def __init__(self, directories, transform=None, train=True):

        self.transform = transform
        self.directories = directories
        self.json_ids = []
        self.train = train

        count = 0
            
        for dir in directories:
            for x in os.listdir(dir):

                if ".json" in x and x != "meta.json":

                    count += 1
                    if self.train and count % 10 == 0:
                        continue
                    if not self.train and count % 10 != 0:
                        continue

                    self.json_ids.append(os.path.join(dir, x))

    def __len__(self):
        return len(self.json_ids)

    def __getitem__(self, idx):

        json_id = self.json_ids[idx]

        with open(json_id) as json:
            j = jsonload(json)

            label = np.array([j['steer'], j['throttle']])
            image_name = '/'.join(json_id.split('/')[:-1]) + '/image_' + json_id.split('/')[-1][5:-5] + '.npy'

        image = np.load(image_name)
        print(image.shape)

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label)

        sample = (image, label.float())

        return sample
