import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

dataset_path = "dataset/first_lap_datasets/big_dataset/rrl1/"
frag_dataset_path = "dataset/first_lap_datasets/big_dataset/fragmented2_rrl1/"


imgs = glob.glob(dataset_path + "*.npy")
if (len(imgs) == 0):
    print("nothing in dataset.")
    exit()

print(f'There are {len(imgs)} images in dataset.')

THRESHOLD = 210
HORIZON = 50

for img in imgs:

    np_img = np.load(img)

    hls_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2HLS)
    lum_img = hls_img[:,:,1]

    frag_img = cv2.inRange(lum_img, 215, 255)

    cropped_img = frag_img
    cropped_img[:50,:] = 0

    final_img = cv2.merge((cropped_img, cropped_img, cropped_img))
    print(final_img.flatten())

    #np.save(frag_dataset_path + img.split('/')[-1], final_img)
    print(f'image {img.split("/")[-1]} saved.')

    img2 = np.load(img)

    img_grey = 0.299 * img2[:,:,0] +  0.587 * img2[:,:,2] +  0.114 * img2[:,:,2] 
    #axs[0].imshow(img_grey, cmap='gray', vmin=0, vmax=255)

    img_grey[:HORIZON,:] = 0

    rows, cols = img_grey.shape

    """
    for i in range(rows - 1, HORIZON - 1, -1):
        mask = img_grey[i,:] > THRESHOLD
        if mask.sum() > 18:
            img_grey[i,:] = 0
    """

    for i in range(rows - 1, HORIZON - 1, -1):

        found = False

        j = cols // 2
        while j < cols - 1:

            if (found == False and img_grey[i,j] > THRESHOLD) : #white

                found = True

                if j < cols - 1:
                    j += 1

                while (img_grey[i,j] > THRESHOLD - 30) :

                    if j < cols - 1 :
                        j += 1
                    else:
                        break

            else:
                img_grey[i,j] = 0

            j += 1

        found = False
        j = cols // 2
        while j > 0:

            if (found == False and img_grey[i,j] > THRESHOLD) : #white

                found = True

                if j == 0 :
                    continue

                j -= 1


                while (img_grey[i,j] > THRESHOLD - 30) :

                    if j > 0 :
                        j -= 1
                    else:
                        break

            else:
                img_grey[i,j] = 0
            j -= 1
    """

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(final_img)
    axs[1].imshow(img_grey)
    plt.show()
    """

    final_img = cv2.merge((img_grey, img_grey, img_grey))
    print(final_img.flatten() > 0)
    np.save(frag_dataset_path + img.split('/')[-1], final_img)
