import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import random
import loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA = ['../../dataset/robo_racing_league_1']

BATCH_SIZE = 5
GREYSCALE = True

if GREYSCALE:
    data = loader.DataGenerator(DATA, (120, 160), 1, batch_size=BATCH_SIZE, grayscale=True)
else:
    data = loader.DataGenerator(DATA, (120, 160), 3, batch_size=BATCH_SIZE)

batch = data[0]
labels = []
images = []

# Before everything: working with float values [0, 1]:
for i in range(BATCH_SIZE):
    labels.append(batch[1][i])
    images.append(batch[0][i])

f, xyarr = plt.subplots(1, BATCH_SIZE)

for i in range(BATCH_SIZE):
    if GREYSCALE:
        xyarr[i].imshow(images[i], cmap='gray')
    else:
        xyarr[i].imshow(images[i])
    xyarr[i].title.set_text(str(labels[i]))

plt.show()
