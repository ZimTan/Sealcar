### Configuration file for model training ###

#IMAGE_DIM = (120, 160, 1)
IMAGE_DIM = (480, 640, 1)

DATASETS_SEG_PATH = ["datasets/23-14-47-853175/frag", "datasets/23-14-52-400121/frag",
                    "datasets/23-14-57-227476/frag", "datasets/23-15-05-852589/frag"]

DATASETS_PATH = ["datasets/23-14-47-853175", "datasets/23-14-52-400121",
                    "datasets/23-14-57-227476", "datasets/23-15-05-852589"]


MODEL_SAVE_PATH = "models/segnvidia_1"
MODEL_SAVE_PATH_SEG = "models/cnnseg_1"

TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
TEST_SIZE = 0.1

MODEL_NAME = "nvidia-speed"
# available model are: nvidia-speed, lstm

EPOCH_SEG = 10

EPOC_CNN = 30

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

LOSS_FUNC = 'mse'

OPTIMIZER = 'adam'
