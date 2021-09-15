# Config file to store the training configuration variables

# DATASET:
DATASET = [
    '../../dataset/fragmented_rrl1',
    ]

# TRAINING PARAMETERS:
BATCH_SIZE = 128
EPOCH = 10
SHUFFLE = True

# MODEL TYPE:
MODEL_TYPE = 'nvidia_speed'

# HYPERPARAMETERS:
LOSS = 'mse'
LEARNING_RATE = 1e-3

# IMAGE PREPROCESSING:
GRAYSCALE = True
IMAGE_DIMENSION = (120, 160, 1)
