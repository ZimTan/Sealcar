# The controler class is used as interface.
class ControllerInterface:
    def __init__(self, config):
        self.config = config

    def set_state(self, data):
        """ Should get the data that you need """
        pass

    def get_throttle(self) -> float:
        """ Should return the value of throttling """
        pass

    def get_steer(self) -> float:
        """ Should return the value of steering """
        pass

    def stop(self) -> bool:
        """ Should return the boolean that stop the car """
        pass