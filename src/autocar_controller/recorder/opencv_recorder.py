import autocar_controller.recorder.recorder_interface as recorder_interface

import os
import json

import numpy as np
from datetime import datetime

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

# The car interface class is used as interface to communicate with your car.
class OpenCvRecorder(recorder_interface.RecorderInterface):
    def __init__(self, config):
        super().__init__(config)

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        self.image_base_name = 'image_'
        self.data_base_name = 'data_'

        self.counter = 0
        self.capture_on = False

    def _on_press(self, key):
        if key == KeyCode.from_char('r'):
            self.capture_on = not self.capture_on
            print("Capture:", self.capture_on)

        if key == KeyCode.from_char('d'):
            print('J\'ai delete je te jure')

    def capture(self, data, action):
        """ Should allow the user to recorde frame by frame """
        if not self.capture_on:
            return

        # JSON data save
        if not 'speed' in data :
            return
        data_to_save = {'steer': action[0], 'throttle': action[1], 'speed': data['speed']}
        today  = datetime.now().strftime("%H-%M-%f")

        with open(os.path.join(self.config['path'], self.data_base_name + str(self.counter)) + today + '.json', 'w+') as outfile:
            json.dump(data_to_save, outfile)

        # Image numpy array save
        np.save(os.path.join(self.config['path'], self.image_base_name + str(self.counter)) + today + '.npy', data['image'])

        self.counter += 1
