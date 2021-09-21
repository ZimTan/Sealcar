import importlib
import loader
import sys
import os
import shutil
import argparser_training
import keras_callbacks
import datetime
import numpy as np

import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # Retrieve the model path and setup the corresponding config file
    args = argparser_training.argparser()
    config_file = os.path.join(args.path, 'config.py')
    model_file = os.path.join(args.path, 'saved_model.pb')
    model_folder = args.path
    log_file = os.path.join(args.path, 'training_log.csv')

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("GPU activated")
        conf = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Not enough GPU hardware devices available, no GPU acceleration enabled")

    # If the model already exists, we load its config file
    # Else we copy the current config file

    import config

    # Dynamically load the choosen model:
    model_module = importlib.import_module("models." + config.MODEL_TYPE)
    model_name = "".join(list(map(lambda n : n[0].upper() + n[1:], config.MODEL_TYPE.split("_")))) + "Model"
    model_class = getattr(model_module, model_name)

    print(model_file)

    if (config.MODEL_TYPE != 'autoencoder'):
        data = loader.DataGenerator(config.DATASET, (config.IMAGE_DIMENSION[0],
                                                     config.IMAGE_DIMENSION[1]), config.IMAGE_DIMENSION[2], config.BATCH_SIZE, config.GRAYSCALE)

    model = (keras.models.load_model(args.path))

    image = np.load("../../dataset/rrl1/image_114-28-251996.npy")
    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    image = (image / 127.5) - 1
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    import cv2
    print(image.shape)
    image = image * 127.5 + 1
    print("icicicicici")
    cv2.imwrite('test2.jpg', image.reshape((image.shape[1], image.shape[2])))

    res = model.predict(image)
    res = (res * 127.5) + 1
    res = res.reshape((res.shape[1], res.shape[2]))

   # cv2.imwrite('test.jpg', res)
