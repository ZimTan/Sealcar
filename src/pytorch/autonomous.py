import sys
import os
import torch
import torchvision
from torchvision import transforms
import nvidia_speed as nvidia_speed
import seg_nvidia as seg_nvidia

import pyrealsense as pyrs

import socket

#********* INIT ********

UDP_IP = "10.42.0.2"
UDP_PORT = 5001
print("Connect to : ", UDP_IP, UDP_PORT)
sock = socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM) # UDP

serv = pyrs.Service()
for dev in serv.get_devices():
    print(dev)
cam = serv.Device(device_id = 0)#, streams = [pyrs.stream.ColorStream(fps = 10)])

if not cam :
    print("error")
    exit()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Avoid using GPU that can be tricky
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device use is : ", device)


# Load the model that you want to use
path = "pytorch/models/" 

# If the model to load does not exist we throw an error
if not os.path.isdir(path):
    sys.exit(path + ': No such file or directory')

model = seg_nvidia.SegNvidia()
model.load_state_dict(torch.load(path + "segnvidia"))
model.eval()

# ********** LOOP *********

#while True
for i in range(5):
    print("LOOP")

    #Get image
    print("Get image")

    cam.wait_for_frames()
    #Prepossesing
    print("img : " + str(cam.color.shape))
    image = cam.color.copy()
    print("img : " + str(image.shape))
    image = transform(image)
    print(image.shape)

    #Inference
    #result = self.model(image[None, ...])[0]
    result = [0,0]
    print("Result : ", result) 

    msg = f"{result[0]};{result[0]};pilot".encode("utf-8")
    print(msg)

    #Send result to ros car
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    print("Msg send")
    #sleep(1)

cam.stop()
serv.stop()


