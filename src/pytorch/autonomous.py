import sys
import os
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import nvidia_speed as nvidia_speed
import seg_nvidia as seg_nvidia

import pyrealsense as pyrs

import socket
import time

SAVE_IMG = True

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
    #transforms.Resize((120, 160)),
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
print("model is loaded.")
model.eval()
model.to(device)

# ********** LOOP *********
count = 0
time_start = time.time()
print("Loop begin ...")
while True:
#for i in range(50):
    print(count)
    count += 1
    #print("LOOP")

    #Get image
    #print("Get image")
    cam.wait_for_frames()

    #Prepossesing
    image = cam.color.copy()
    #print("img : " + str(image.shape))
    print("coucou")
    image = transform(image)
    if SAVE_IMG:
        save_image(image, "imgs/img-" + str(count) + ".png")
    #print(image.shape)
    #print(image)

    #Inference
    print("Inference")
    with torch.no_grad():
        result = model(image[None, ...].to(device)).cpu()
    result = result[0]

    throtle = round(result[0].item(), 3) + 0.5
    steer = round(result[1].item(), 3) + 0.5
    print(throtle, steer)

    msg1 = f"{throtle};{steer};pilot".encode("utf-8")
    msg2 = f"{throtle};{steer};pilot;{count}".encode("utf-8")
    print(msg1)

    #Send result to ros car
    sock.sendto(msg1, (UDP_IP, UDP_PORT))
    sock.sendto(msg2, (UDP_IP, UDP_PORT + 1))
    #print("Msg send")

    time_end = time.time()
    print(time_end - time_start)
    print()
    time_start = time_end

cam.stop()
serv.stop()


