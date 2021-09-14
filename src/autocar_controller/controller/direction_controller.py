import autocar_controller.controller.controller_interface as controller_interface

from pynput import keyboard
from pynput.keyboard import Key

class DirectionController(controller_interface.ControllerInterface):
    def __init__(self, config):
        super().__init__(config)
        
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release)
        self.listener.start()

        self.direction = [-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1]
        self.acceleration = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]

        self.direction_index = 5
        self.acceleration_index = 2
        self.KEYBOARD_ESC = False

    def _on_press(self, key):
        if key == Key.right:
            self.direction_index += 1
        if key == Key.left:
            self.direction_index -= 1
        if key == Key.up:
            self.acceleration_index += 1
        if key == Key.down:
            self.acceleration_index -= 1
        if key == Key.esc:
            self.KEYBOARD_ESC = True

    def _on_release(self, key):
        if key == Key.esc:
            self.KEYBOARD_ESC = False

    def get_throttle(self) -> float:
        return self.direction[self.direction_index]

    def get_steer(self) -> float:
        return self.acceleration[self.acceleration_index]

    def stop(self) -> bool:
        return self.KEYBOARD_ESC