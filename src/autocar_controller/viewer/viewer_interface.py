# The car interface class is used as interface to communicate with your car.
class ViewerInterface:
    def __init__(self, config):
        self.config = config

    def watch(self, data):
        """ Should get allow the user to watch frame by frame """
        pass
        
    def on(self):
        """ Should turn on the view """
        pass

    def off(self):
        """ Should turn off the view """
        pass