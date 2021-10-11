### Configuration file for model training ###

IMAGE_DIM = (120, 160, 1)

DATASETS_PATH = ["big_dataset/fragmented_rrl1", "scared_dataset/fragmented_rrl1"]
MODEL_SAVE_PATH = "models/lstm"

TRAIN_SIZE = 0.7
VALID_SIZE = 0.2
TEST_SIZE = 0.1

SHUFFLE_DATA = False

MODEL_NAME = "lstm"
# available model are: nvidia-speed, lstm

EPOCH = 15

BATCH_SIZE = 128
SEQUENCE_SIZE = 15

LEARNING_RATE = 1e-3

LOSS_FUNC = 'cross-entropy'
# available loss function are: mse, cross-entropy

OPTIMIZER = 'adam'
