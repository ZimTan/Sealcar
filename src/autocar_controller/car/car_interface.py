# The car interface class is used as interface to communicate with your car.
class CarInterface:
    def __init__(self, config):
        self.config = config
        
    def get_data(self) -> dict:
        """ Should return the value of throttling """
        pass

    def send_action(self):
        """ Should return the value of steering """
        pass

    def pause(self):
        """ Should return the boolean that stop the car """
        pass

    def stop(self):
        """ Should return the boolean that stop the car """
        pass