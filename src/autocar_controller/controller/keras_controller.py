import autocar_controller.controller.controller_interface as controller_interface

import os
import sys

import numpy as np
import config as conf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from pynput import keyboard
from pynput.keyboard import Key

# The keras Controller Class.
class KerasController(controller_interface.ControllerInterface):
    def __init__(self, config):
        super().__init__(config)
        self.throttle = None
        self.steer = None
        self.model = None
#        self.default_image = None
#        self.default_speed = None

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        self.KEYBOARD_ESC = False

        # Avoid using GPU that can be tricky
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Load the model that you want to use
        path = config['path']

        # If the model to load does not exist we throw an error
        if not os.path.isdir(path):
            sys.exit(path + ': No such file or directory')

        self.model = keras.models.load_model(path)

    def _on_press(self, key):
        if key == Key.esc:
            self.KEYBOARD_ESC = True

    def __transform_grayscale__(self, image):
        im = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return (im / 127.5) - 1

    def __transform_color__(self, image):
        return (image / 127.5) - 1

    def set_state(self, data):
        #        if ('image' in data.keys()):
        #            self.default_image = data['image']
        #        else:
        #            data['image'] = self.default_image
        #
        #        if ('speed' in data.keys()):
        #            self.default_speed = data['speed']
        #        else:
        #            data['speed'] = self.default_speed

        image = tf.convert_to_tensor(np.array([np.expand_dims(self.__transform_grayscale__(data['image']), 2)]), dtype=tf.float32)
        speed = tf.convert_to_tensor(np.array([np.array([data['speed']])]), dtype=tf.float32)

        result = self.model.call((image, speed))[0]

        self.steer = result[0]
        self.throttle = result[1]

    def get_throttle(self) -> float:
        return self.throttle

    def get_steer(self) -> float:
        return self.steer

    def stop(self) -> bool:
        return self.KEYBOARD_ESC
