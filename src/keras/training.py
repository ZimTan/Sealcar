import importlib
import loader
import sys
import os
import shutil
import argparser_training
import keras_callbacks

import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # Retrieve the model path and setup the corresponding config file
    args = argparser_training.argparser()
    config_file = os.path.join(args.path, 'config.py')
    model_file = os.path.join(args.path, 'model.h5')
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
    if os.path.isfile(config_file) and not args.create:
        sys.path.insert(0, args.path)
        model_loaded = True
        print("Model Loaded")

    else:
        shutil.rmtree(args.path, ignore_errors=True)

        os.makedirs(args.path, exist_ok=True)
        shutil.copyfile('config.py', config_file)

        model_loaded = False
        print("Model Created")

    import config

    # Dynamically load the choosen model:
    model_module = importlib.import_module("models." + config.MODEL_TYPE)
    model_name = "".join(list(map(lambda n : n[0].upper() + n[1:], config.MODEL_TYPE.split("_")))) + "Model"
    model_class = getattr(model_module, model_name)

    model = (keras.models.load_model(model_file) if model_loaded else model_class(config.IMAGE_DIMENSION, 2))

    model.compile(loss=config.LOSS, optimizer=tf.keras.optimizers.Adam(lr=config.LEARNING_RATE))

    data = loader.DataGenerator(config.DATASET, (config.IMAGE_DIMENSION[0],
        config.IMAGE_DIMENSION[1]), config.IMAGE_DIMENSION[2], config.BATCH_SIZE, config.GRAYSCALE)

    callbacks = [
        keras_callbacks.ModelCheckpointTFFormat(path=model_folder),
        keras.callbacks.CSVLogger(log_file, separator=",", append=True)
    ]

    history = model.fit(data, epochs=config.EPOCH, shuffle=config.SHUFFLE, workers=4, use_multiprocessing=args.thread, verbose=1, callbacks=callbacks)
