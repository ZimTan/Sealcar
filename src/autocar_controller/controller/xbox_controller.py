import autocar_controller.controller.controller_interface as controller_interface

import pygame

class XboxController(controller_interface.ControllerInterface):
    def __init__(self, config):
        super().__init__(config)

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise Exception("No valid controller detected")

        self.joystick = pygame.joystick.Joystick(0)
        print("Controller Name: ", self.joystick.get_name())

    def set_state(self, data):
        pass

    def get_throttle(self) -> float:
        axis_forward = (self.joystick.get_axis(5) + 1) / 2
        axis_backward = -(self.joystick.get_axis(2) + 1) / 2
        return axis_forward + axis_backward + self.config['fix_steer']

    def get_steer(self) -> float:
        axis = self.joystick.get_axis(0)
        if axis > -self.config['dead_zone'] and axis < self.config['dead_zone']:
            return 0
        elif axis > 0:
            return axis - self.config['dead_zone']
        else:
            return axis + self.config['dead_zone']

    def stop(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                button_b = self.joystick.get_button(1)
                if button_b == 1:
                    return True
