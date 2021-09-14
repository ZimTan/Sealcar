# The recorder interface class is used as interface to create a recorder.
class RecorderInterface:
    def __init__(self, config):
        self.config = config

    def capture(self, data, action):
        """ Should allow the user to recorde frame by frame """
        pass