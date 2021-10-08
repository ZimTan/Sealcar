### Configuration file for model training ###

IMAGE_DIM = (120, 160, 1)

DATASETS_PATH = ["first_lap_datasets/big_dataset/fragmented_rrl1/", "first_lap_datasets/scared_dataset/fragmented_rrl1/"]
MODEL_SAVE_PATH_SEG = "models/cnnseg"
MODEL_SAVE_PATH = "models/segnvidia"

TRAIN_SIZE = 0.7
VALID_SIZE = 0.2
TEST_SIZE = 0.1

MODEL_NAME = "nvidia-speed"
# available model are: nvidia-speed, lstm

EPOCH = 1

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

LOSS_FUNC = 'mse'

OPTIMIZER = 'adam'
