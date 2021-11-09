### Configuration file for model training ###

#IMAGE_DIM = (120, 160, 1)
IMAGE_DIM = (480, 640, 1)

DATASETS_SEG_PATH = ["dataset_real/23-14-47-853175/frag"] 

DATASETS_PATH = ["dataset_real/23-14-47-853175"]


MODEL_SAVE_PATH = "src/pytorch/models/segnvidia"
MODEL_SAVE_PATH_SEG = "src/pytorch/models/cnnseg"

TRAIN_SIZE = 0.7
VALID_SIZE = 0.2
TEST_SIZE = 0.1

MODEL_NAME = "nvidia-speed"
# available model are: nvidia-speed, lstm

EPOCH_SEG = 10

EPOC_CNN = 30

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

LOSS_FUNC = 'mse'

OPTIMIZER = 'adam'
