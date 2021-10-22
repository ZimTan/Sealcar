import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

dataset_path = "/home/tanz/Bureau/data_sealcar/"
frag_dataset_path = "/home/tanz/Bureau/data_sealcar/frag/"

imgs = []
LEFT = 0
RIGHT = 1
#for img in os.listdir(dataset_path):
#    imgs.append(dataset_path + img)

imgs = glob.glob(dataset_path + "*.npy")
if (len(imgs) == 0):
    print("nothing in dataset.")
    exit()

print(f'There are {len(imgs)} images in dataset.')

THRESHOLD = 200
HORIZON = 180
#print(imgs[0])

def update_threshold(image, i, j, cols):
    result = 0
    j += 1
    while (j < cols - 1):
        if (image[i,j] > result):
            result = image[i,j]
        j += 1

    return result - 10;

for img in imgs:

    print(img)
    np_img = np.load(img)
    #np_img = cv2.imread(img)

    hls_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2HLS)
    lum_img = hls_img[:,:,1]

    frag_img = cv2.inRange(lum_img, 215, 255)

    cropped_img = frag_img
    cropped_img[:50,:] = 0

    final_img = cv2.merge((cropped_img, cropped_img, cropped_img))
    #print(final_img.flatten())

    #np.save(frag_dataset_path + img.split('/')[-1], final_img)
    #print(f'image {img.split("/")[-1]} saved.')

    img2 = np.load(img)
   # img2 = cv2.imread(img)

    img_grey = 0.299 * img2[:,:,0] +  0.587 * img2[:,:,2] +  0.114 * img2[:,:,2]
   # plt.imshow(img_grey, cmap='gray', vmin=0, vmax=255)
    #plt.show()

    img_grey[:HORIZON,:] = 0

    rows, cols = img_grey.shape

    """
    for i in range(rows - 1, HORIZON - 1, -1):
        mask = img_grey[i,:] > THRESHOLD
        if mask.sum() > 18:
            img_grey[i,:] = 0
    """

    for i in range(HORIZON - 1):
        for j in range(cols):
            img_grey[i,j] = 0

    for i in range(350, rows - 1):
        for j in range(cols):
            img_grey[i,j] = 0

    for i in range(HORIZON, 350):

        found = False
        brightness_sum = 0

        j = cols // 2

        THRESHOLD = update_threshold(img_grey, i, j, cols)

        while j < cols:

            if (found == False and img_grey[i,j] > THRESHOLD): #white
                found = True
                if j < cols - 1:
                    j += 1

                while (j < cols and img_grey[i,j] > THRESHOLD - 5):
                    brightness_sum += img_grey[i,j]
                    j += 1
                if (brightness_sum > THRESHOLD * 60):
                    for k in range(cols // 2, cols - 1):
                        img_grey[i,k] = 0


            else:
                img_grey[i,j] = 0

            j += 1

        found = False
        brightness_sum = 0
        j = cols // 2

        THRESHOLD = update_threshold(img_grey, i, 0, j)

        while j > 0:
            if (found == False and img_grey[i,j] > THRESHOLD) : #white
                found = True
                if j > 0 :
                    j -= 1

                while (j > 0 and img_grey[i,j] > THRESHOLD - 5):
                    brightness_sum += img_grey[i,j]
                    j -= 1
                if (brightness_sum > THRESHOLD * 60):
                    for k in range(cols // 2):
                        img_grey[i,k] = 0

            else:
                img_grey[i,j] = 0
            j -= 1







    final_img = cv2.merge((img_grey, img_grey, img_grey))
    #print(final_img.flatten() > 0)
    np.save(frag_dataset_path + img.split('/')[-1], final_img)

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(final_img)
    axs[1].imshow(np_img)
    plt.show()
