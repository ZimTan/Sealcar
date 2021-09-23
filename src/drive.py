import numpy
import time

import autocar_config

# Get the controller interface
import autocar_controller.controller.controller_interface as controller_interface

# Get the car interface
import autocar_controller.car.car_interface as car_interface

# Get the viewer interface
import autocar_controller.viewer.viewer_interface as viewer_interface

# Get the recorder interface
import autocar_controller.recorder.recorder_interface as recorder_interface

# Car:
# Import your own interface to communicate with your car:
import autocar_controller.car.client as client

# Viewer:
# Import your own controller if you need
import autocar_controller.viewer.opencv_viewer as viewer_lib
#import autocar_controller.viewer.console_viewer as viewer_lib

# Recorder:
# Import your own recorder if you need
import autocar_controller.recorder.opencv_recorder as opencv_recorder

# Retrieving the call arguments
import argparser_drive

args = argparser_drive.argparser()

# driver config:
VIEW_ON = True

# Controller:
if args.mode == 'keyboard':
    import autocar_controller.controller.keyboard_controller as keyboard_controller
    CONTROLLER = keyboard_controller.KeyboardController

elif args.mode == 'keras':
    import autocar_controller.controller.keras_controller as keras_controller
    CONTROLLER = keras_controller.KerasController

elif args.mode == 'direction':
    import autocar_controller.controller.direction_controller as direction_controller
    CONTROLLER = direction_controller.DirectionController

elif args.mode == 'xbox':
    import autocar_controller.controller.xbox_controller as xbox_controller
    CONTROLLER = xbox_controller.XboxController

elif args.mode == 'pytorch':
    import autocar_controller.controller.pytorch_controller as pytorch_controller
    CONTROLLER = pytorch_controller.PytorchController

CAR = client.UnitySimulationClient
VIEWER = viewer_lib.OpenCvViewer
RECORDER = opencv_recorder.OpenCvRecorder

# Check up:
# We need to be sur that the controller respect the Controller Interface
if not issubclass(CONTROLLER, controller_interface.ControllerInterface):
    raise Exception('Your controller class should respect the Controller interface')

# We need to be sur that the car respect the Car Interface
if not issubclass(CAR, car_interface.CarInterface):
    raise Exception('Your car class should respect the Car interface')

# We need to be sur that the viwer respect the Viewer Interface
if not issubclass(VIEWER, viewer_interface.ViewerInterface):
    raise Exception('Your viewer class should respect the Viewer interface')

# We need to be sur that the viwer respect the Viewer Interface
if not issubclass(RECORDER, recorder_interface.RecorderInterface):
    raise Exception('Your recorder class should respect the Recorder interface')

# Initialisation:
# Set up controler:
controller = CONTROLLER(autocar_config.controller_config)

# Set up car:
car = CAR(autocar_config.car_config)

# Set up viewer:
viewer = VIEWER(autocar_config.viewer_config)

# Set up recorder:
recorder = RECORDER(autocar_config.recorder_config)
#recorder.capture_on = True;

# Start driving:
action = numpy.array([0, 0.0])

start = time.time()
nb_frame = 0

while not controller.stop():
    data = car.get_data()

    controller.set_state(data)

    action[0] = controller.get_steer()
    action[1] = controller.get_throttle()

    recorder.capture(data, action)

    car.send_action(action)

    viewer.watch(data)

    nb_frame += 1

end = time.time()
print("fps:", int(nb_frame / int(end - start)))

# When everything done
# release the viewer
viewer.off()
# turn off car
car.stop()
