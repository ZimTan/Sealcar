import os
import random

from tensorflow import keras

import tensorflow as tf
import numpy as np

from json import load as jsonload

import preprocessing_functions

class DataGenerator(keras.utils.Sequence):
    def __init__(self, directories, dim, n_channels, batch_size=32, grayscale=False, autoencoder=False):
        self.directories = directories
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.grayscale = grayscale
        self.autoencoder = autoencoder

        if self.grayscale:
            self.n_channels = 1

        self.json_ids = []

        for dir in directories:
          self.json_ids += [os.path.join(dir, x) for x in os.listdir(dir) if ".json" in x and x != "meta.json"]

    def __len__(self):
        return len(self.json_ids) // self.batch_size

    def transform_grayscale(self, image):
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return (image / 127.5) - 1

    def transform_color(self, image):
        return (image / 127.5) - 1

    def transform_random(self, image, angle):
        image = preprocessing_functions.random_threashold(image, 0.2)
        image = tf.keras.preprocessing.image.random_brightness(image, (0.85, 1.15))
        image = preprocessing_functions.random_invert_img(image, 0.3)
        #image = preprocessing_functions.random_cache(image, 1)
        image = preprocessing_functions.add_noise(image, 50)
        image, angle = preprocessing_functions.random_flip(image, angle)
        angle = preprocessing_functions.random_angle_noise(angle)
        return image, angle

    def __getitem__(self, index):
        image_batch = np.empty((self.batch_size, *self.dim, self.n_channels))
        speed_batch = np.empty((self.batch_size, 1))
        args_batch = []
        label_batch = np.empty((self.batch_size, 2), dtype=float)
        f_index = self.json_ids[index * self.batch_size:(index + 1) * self.batch_size]

        for i, json_id in enumerate(f_index):
            image_name = ''
            speed = 0.0

            with open(json_id) as json:
                j = jsonload(json)
                label_batch[i] = np.array([j['steer'], j['throttle']])
                speed = j['speed']
                image_name = '/'.join(json_id.split('/')[:-1]) + '/image_' + json_id.split('/')[-1][5:-5] + '.npy'

            image = np.load(image_name)
            image, label_batch[i] = self.transform_random(image, label_batch[i])

            if self.grayscale:
                image_batch[i] = self.transform_grayscale(np.expand_dims(image, 2))
                speed_batch[i] = speed
                #image_batch[0].append(self.transform_grayscale(np.expand_dims(image, 2)))
                #image_batch[1].append(speed)
            else:
                image_batch[i] = self.transform_color(image)

        args_batch.append(image_batch)
        args_batch.append(speed_batch)

        if self.autoencoder == True:
            return image_batch, image_batch

        return args_batch, label_batch

    def on_epoch_end(self):
        np.random.shuffle(self.json_ids)
