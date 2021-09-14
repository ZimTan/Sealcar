import autocar_controller.viewer.viewer_interface as viewer_interface

import os
import numpy as np

# The car interface class is used as interface to communicate with your car.
class ConsoleViewer(viewer_interface.ViewerInterface):
    def __init__(self, config):
        super().__init__(config)
        self.clear = lambda: os.system('clear')

    def watch(self, data):
        """ Should get allow the user to watch frame by frame """
        image = data['image']
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(int)
        image[image >= 100] = 1
        image[image < 100] = 0
        for i in image:
            for j in i:
                print(j, end='')
            print()
        self.clear()
        
    def on(self):
        """ Should turn on the view """
        pass

    def off(self):
        """ Should turn off the view """
        pass