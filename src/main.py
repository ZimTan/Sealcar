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
#recorder.capture_on = True;

# Start driving:
action = numpy.array([0, 0.0])

HardCode_steer = [0 for i in range(17)]
HardCode_throttle = [1 for i in range(15)]

HardCode_steer += [1 for i in range(15)]
HardCode_throttle += [0.6 for i in range(17)]

HardCode_steer += [0 for i in range(5)]
HardCode_throttle += [1 for i in range(5)]

HardCode_steer += [-1 for i in range(8)]
HardCode_throttle += [0.6 for i in range(8)]

HardCode_steer += [0 for i in range(11)]
HardCode_throttle += [1 for i in range(6)]
HardCode_throttle += [0.6 for i in range(3)]
HardCode_throttle += [0.3 for i in range(2)]

HardCode_steer += [-1 for i in range(18)]
HardCode_throttle += [0.5 for i in range(18)]

HardCode_steer += [0 for i in range(8)]
HardCode_throttle += [1 for i in range(5)]
HardCode_throttle += [0.6 for i in range(2)]
HardCode_throttle += [0.3 for i in range(2)]

HardCode_steer += [1 for i in range(16)]
HardCode_throttle += [0.4 for i in range(16)]


HardCode_steer += [0 for i in range(34)]
HardCode_throttle += [1 for i in range(26)]
HardCode_throttle += [0.6 for i in range(4)]
HardCode_throttle += [0.3 for i in range(2)]
HardCode_throttle += [0.1 for i in range(2)]

HardCode_steer += [1 for i in range(12)]
HardCode_throttle += [0.3 for i in range(12)]

HardCode_steer += [0 for i in range(2)]
HardCode_throttle += [1 for i in range(2)]
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
