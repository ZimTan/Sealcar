import autocar_controller.controller.controller_interface as controller_interface

from pynput import keyboard
from pynput.keyboard import Key

class KeyboardController(controller_interface.ControllerInterface):
    def __init__(self, config):
        super().__init__(config)
        
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release)
        self.listener.start()

        self.KEYBOARD_RIGHT = 0
        self.KEYBOARD_LEFT = 0
        self.KEYBOARD_UP = 0
        self.KEYBOARD_DOWN = 0
        self.KEYBOARD_ESC = False

    def _on_press(self, key):
        if key == Key.right:
            self.KEYBOARD_RIGHT = 1
        if key == Key.left:
            self.KEYBOARD_LEFT = -1
        if key == Key.up:
            self.KEYBOARD_UP = 1
        if key == Key.down:
            self.KEYBOARD_DOWN = -1
        if key == Key.esc:
            self.KEYBOARD_ESC = True

    def _on_release(self, key):
        if key == Key.right:
            self.KEYBOARD_RIGHT = 0
        if key == Key.left:
            self.KEYBOARD_LEFT = 0
        if key == Key.up:
            self.KEYBOARD_UP = 0
        if key == Key.down:
            self.KEYBOARD_DOWN = 0
        if key == Key.esc:
            self.KEYBOARD_ESC = False

    def get_right(self) -> float:
        return self.KEYBOARD_RIGHT

    def get_left(self) -> float:
        return self.KEYBOARD_LEFT

    def get_up(self) -> float:
        return self.KEYBOARD_UP

    def get_down(self) -> float:
        return self.KEYBOARD_DOWN

    def get_throttle(self) -> float:
        return self.KEYBOARD_UP + self.KEYBOARD_DOWN

    def get_steer(self) -> float:
        return self.KEYBOARD_RIGHT + self.KEYBOARD_LEFT

    def stop(self) -> bool:
        return self.KEYBOARD_ESC
