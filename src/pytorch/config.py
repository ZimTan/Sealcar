### Configuration file for model training ###

IMAGE_DIM = (120, 160, 1)

DATASETS_PATH = ["dataset/fragmented_rrl1/"]
MODEL_SAVE_PATH = "models/nvidia-speed"

TRAIN_SIZE = 0.7
VALID_SIZE = 0.2
TEST_SIZE = 0.1

MODEL_NAME = "nvidia-speed"
# available model are: nvidia-speed, lstm

EPOCH = 15

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

LOSS_FUNC = 'mse'

OPTIMIZER = 'adam'
