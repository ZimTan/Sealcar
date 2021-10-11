### Configuration file for model training ###

IMAGE_DIM = (120, 160, 1)

DATASETS_SEG_PATH = [("dataset/first_lap_datasets/scared_dataset/rrl1", "dataset/first_lap_datasets/scared_dataset/fragmented_rrl1"), ("dataset/first_lap_datasets/big_dataset/rrl1", "dataset/first_lap_datasets/big_dataset/fragmented_rrl1")]

DATASETS_PATH = ["dataset/first_lap_datasets/scared_dataset/rrl1", "dataset/first_lap_datasets/big_dataset/rrl1"]
MODEL_SAVE_PATH_SEG = "src/pytorch/models/cnnseg"
MODEL_SAVE_PATH = "src/pytorch/models/segnvidia"

TRAIN_SIZE = 0.7
VALID_SIZE = 0.2
TEST_SIZE = 0.1

MODEL_NAME = "nvidia-speed"
# available model are: nvidia-speed, lstm

EPOCH_SEG = 1
EPOCH_SEG = 10

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

LOSS_FUNC = 'mse'

OPTIMIZER = 'adam'
