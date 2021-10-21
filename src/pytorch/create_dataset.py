import pyrealsense as pyrs

import socket
import time

import json

# Make it work for Python 2+3 and with Unicode
import io
import os

from datetime import datetime
import numpy as np

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

#********* INIT ********

UDP_IP = "10.42.0.1"
UDP_PORT = 5001
print("Connect to : ", UDP_IP, UDP_PORT)
sock = socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

serv = pyrs.Service()
for dev in serv.get_devices():
    print(dev)

cam = serv.Device(device_id = 0)#, streams = [pyrs.stream.ColorStream(fps = 10)])

if not cam :
    print("error")
    exit()

index = 0

name_directory = datetime.now().strftime("%d-%H-%M") + "/"

path = "dataset/" + name_directory

os.mkdir(path)

while True:

    #Get image
    cam.wait_for_frames()

    image = cam.color.copy()

    msg1, _ = sock.recvfrom(1024)
    msg2, _ = sock.recvfrom(1024)
    st = msg1.decode().split(';')[1]
    th = msg2.decode().split(';')[1]

    # Define data
    data = {"steer": float(st), "throttle": float(th), "speed": 0.0}

    # Write JSON file
    with io.open(path + str(index) + '.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data)
        outfile.write(to_unicode(str_))
        np.save(path + str(index) + '.npy', image)

cam.stop()
serv.stop()


