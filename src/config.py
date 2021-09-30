# Config file to store the training configuration variables

# Model to test new normalisation grayscale method: (im / 127.5) - 1

# DATASET
DATASET = [
    '../../dataset/data_analisys',
    ]
BATCH_SIZE = 128
EPOCH = 20

# MODEL TYPE
MODEL_TYPE = 'lstm'

# IMAGE DIMENSION
IMAGE_DIMENSION = (120, 160, 1)

# GRAYSCALE
GRAYSCALE = True
