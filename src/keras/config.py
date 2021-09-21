# Config file to store the training configuration variables

# DATASET:
DATASET = [
    '../../dataset/rrl1',
    ]

# TRAINING PARAMETERS:
BATCH_SIZE = 18
EPOCH = 35
SHUFFLE = True

# MODEL TYPE:
MODEL_TYPE = 'autoencoder' # 'nvidia_speed'

# HYPERPARAMETERS:
LOSS = 'mse'
LEARNING_RATE = 1e-3

# IMAGE PREPROCESSING:
GRAYSCALE = True
IMAGE_DIMENSION = (120, 160, 1)
