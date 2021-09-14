import numpy
import time

import autocar_config
import autocar_controller.car.car_interface as car_interface
import autocar_controller.car.client as client
import autocar_controller.recorder.recorder_interface as recorder_interface
import autocar_controller.recorder.opencv_recorder as opencv_recorder

# Initialisation:
CAR = client.UnitySimulationClient
RECORDER = opencv_recorder.OpenCvRecorder

# Set up car:
car = CAR(autocar_config.car_config)
recorder = RECORDER(autocar_config.recorder_config)
recorder.capture_on = True;

# Start driving:
action = numpy.array([0, 0.0])

HardCode_steer = [0 for i in range(16)]
HardCode_throttle = [1 for i in range(14)]

HardCode_steer += [1 for i in range(15)]
HardCode_throttle += [0.6 for i in range(17)]

HardCode_steer += [0 for i in range(8)]
HardCode_throttle += [1 for i in range(8)]

counter = 0

while counter < len(HardCode_steer):
    data = car.get_data()

    action[0] = HardCode_steer[counter]#controller.get_steer()
    action[1] = HardCode_throttle[counter]#controller.get_throttle()

    recorder.capture(data, action)

    car.send_action(action)

    time.sleep(0.2)
    counter += 1

# When everything done
# turn off car
car.stop()
