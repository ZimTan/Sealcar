import sys
import os
import torch
import torchvision
from torchvision import transforms
import pytorch.nvidia_speed as nvidia_speed
import pytorch.seg_nvidia as seg_nvidia

#********* INIT ********

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Avoid using GPU that can be tricky
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the model that you want to use
path = "pytorch/models/" 

# If the model to load does not exist we throw an error
if not os.path.isdir(path):
    sys.exit(path + ': No such file or directory')

model = seg_nvidia.SegNvidia()
model.load_state_dict(torch.load(path + "segnvidia"))
model.eval()

# ********** LOOP *********

while True:

    #Get image

    image = []

    #Prepossesing
    image = self.transform(image)

    #Inference
    result = self.model(image[None, ...])[0]
    print("Result : ", result) 

    #Send result to ros car



