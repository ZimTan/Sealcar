import autocar_controller.viewer.viewer_interface as viewer_interface

import cv2

# The car interface class is used as interface to communicate with your car.
class OpenCvViewer(viewer_interface.ViewerInterface):
    def __init__(self, config):
        super().__init__(config)

    def watch(self, data):
        """ Should get allow the user to watch frame by frame """
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.config['screen_size'])
        #cv2.imshow('frame',image)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   return False
        
    def on(self):
        """ Should turn on the view """
        pass

    def off(self):
        """ Should turn off the view """
        #cap.release()
        cv2.destroyAllWindows()
