import autocar_controller.controller.controller_interface as controller_interface

import os
import sys

import numpy as np
import config as conf

import torch
import torchvision
from torchvision import transforms
import pytorch.nvidia_speed as nvidia_speed
import pytorch.lstm as lstm

from pynput import keyboard
from pynput.keyboard import Key

import cv2

# The keras Controller Class.
class PytorchController(controller_interface.ControllerInterface):
    def __init__(self, config):
        super().__init__(config)
        self.throttle = None
        self.steer = None
        self.model = None
#        self.default_image = None
#        self.default_speed = None

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.KEYBOARD_ESC = False

        # Avoid using GPU that can be tricky
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Load the model that you want to use
        path = config['path']

        # If the model to load does not exist we throw an error
        if not os.path.isdir(path):
            sys.exit(path + ': No such file or directory')

        #self.model = nvidia_speed.NvidiaSpeed()
        self.model = lstm.LSTM()
        self.model.load_state_dict(torch.load(path + "/" + conf.MODEL_TYPE))
        self.model.eval()

    def _on_press(self, key):
        if key == Key.esc:
            self.KEYBOARD_ESC = True

    def __transform_grayscale__(self, image):
        im = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return (im / 127.5) - 1

    def segmentation(self, image):

        hls_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        lum_img = hls_img[:,:,1]

        frag_img = cv2.inRange(lum_img, 215, 255)

        cropped_img = frag_img
        cropped_img[:50,:] = 0

        #final_img = cv2.merge((cropped_img, cropped_img, cropped_img))
        return cropped_img


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

        image = self.segmentation(data['image'])
        image = self.transform(image)

        result = self.model(image[None, ...])[0]
        print("Result : ", result) 

        self.steer = result[0]
        self.throttle = result[1]

    def get_throttle(self) -> float:
        return self.throttle

    def get_steer(self) -> float:
        return self.steer

    def stop(self) -> bool:
        return self.KEYBOARD_ESC
