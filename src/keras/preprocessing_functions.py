import tensorflow as tf
import random
import numpy as np

def add_noise(img, var):
    '''Add random noise to an image'''
    deviation = var*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    return np.clip(img, 0., 255.)

def random_flip(image, angle, p=0.5):
    if random.random() < p:
        image = np.fliplr(image)
        angle[0] = -angle[0]
    return image, angle

def random_invert_img(image, p=0.5):
  if random.random() < p:
    image = (255-image)
  return image

def random_angle_noise(angle):
  (e1, e2) = ((random.random() - 0.5) / 1e6, (random.random() - 0.5) / 1e6)
  angle[0] += e1
  angle[1] += e2
  return angle

def random_cache(image, p=0.5):
  #if random.random() < p:
    #print(image.shape)

    #height = np.random.randint(10,40)
    #width = np.random.randint(15,60)
    #cache = np.array([np.random.randint(0,1)]*height*width).reshape(height,width)
    #x = np.random.randint(0,15)
    #y = np.random.randint(0,25)

    #image[x:x+height, y:y+width, 0] = cache[:]
    #image[x:x+height, y:y+width, 1] = cache[:]
    #image[x:x+height, y:y+width, 2] = cache[:]

    #image[0:30, 0:image.shape[0], 0] = np.random.randint(0,255)
    #image[0:30, 0:image.shape[0], 1] = np.random.randint(0,255)
    #image[0:30, 0:image.shape[0], 2] = np.random.randint(0,255)
  return image

def random_threashold(image, p=0.5):
  if random.random() < p:
    image[image <= 230] = 0
    image[image > 230] = 255
  return image