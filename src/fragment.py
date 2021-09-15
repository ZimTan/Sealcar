import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

dataset_path = "dataset/rrl1/"
frag_dataset_path = "dataset/fragmented_rrl1/"


imgs = glob.glob(dataset_path + "*.npy")
if (len(imgs) == 0):
    print("nothing in dataset.")
    exit()

print(f'There are {len(imgs)} images in dataset.')


for img in imgs:


    np_img = np.load(img)

    hls_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2HLS)
    lum_img = hls_img[:,:,1]

    frag_img = cv2.inRange(lum_img, 215, 255)

    cropped_img = frag_img
    cropped_img[:50,:] = 0

    final_img = cv2.merge((cropped_img, cropped_img, cropped_img))

    np.save(frag_dataset_path + img.split('/')[-1], final_img)
    print(f'image {img.split("/")[-1]} saved.')
