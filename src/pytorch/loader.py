import os
import random
import torch

from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from json import load as jsonload

class AddNoise:
   # def __init__(self):
    def __call__(self, x):
        mask = np.ma.masked_where(x<=220, x).mask
        x[mask] = np.random.randint(random.randint(170,255), size=x.shape)[mask] 
        return x

class RemoveTop:
   # def __init__(self):
    def __call__(self, x):
        return x[40:,:]

class SegDataSet(Dataset):

    def __init__(self, directories, transform=None, train=True):

        self.transform = transform
        self.directories = directories
        self.json_ids = []
        self.train = train

        self.input_transforms = transforms.Compose([
            AddNoise(),
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

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
            image_name = '/'.join(json_id.split('/')[:-1]) + '/image_' + json_id.split('/')[-1][5:-5] + '.npy'

        image = np.load(image_name)

        if self.input_transforms:
            in_image = self.input_transforms(np.copy(image))

        if self.transform:
            image = self.transform(image)

        """
        f, ax = plt.subplots(2)
        ax[0].imshow(in_image.squeeze(dim=0), cmap='gray')
        ax[1].imshow(image.squeeze(dim=0), cmap='gray')
        plt.show()
        """

        sample = (in_image, image)

        return sample


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

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label)

        """
        f, ax = plt.subplots(2)
        ax[0].imshow(in_image.squeeze(dim=0), cmap='gray')
        ax[1].imshow(image.squeeze(dim=0), cmap='gray')
        plt.show()
        """

        sample = (image, label.float())

        return sample

