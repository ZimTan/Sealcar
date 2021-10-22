### Configuration file for model training ###

#IMAGE_DIM = (120, 160, 1)
IMAGE_DIM = (480, 640, 1)

DATASETS_SEG_PATH = ["dataset_real/21-16-44-867066/frag"] 

DATASETS_PATH = ["dataset_real/21-16-44-867066"]

DATASETS_PATH = ["/home/banner/Sealcar/dataset/21-16-44-867066", "/home/banner/Sealcar/dataset/21-16-51-573448", "/home/banner/Sealcar/dataset/21-17-28-237168", "/home/banner/Sealcar/dataset/21-17-34-062137"]

DATASETS_SEG_PATH = ["/home/banner/Sealcar/dataset/21-16-44-867066/frag", "/home/banner/Sealcar/dataset/21-16-51-573448/frag", "/home/banner/Sealcar/dataset/21-17-28-237168/frag", "/home/banner/Sealcar/dataset/21-17-34-062137/frag"]
#DATASETS_PATH = ["/home/banner/Sealcar/dataset/21-16-44-867066"]


MODEL_SAVE_PATH = "src/pytorch/models/segnvidia"
MODEL_SAVE_PATH_SEG = "src/pytorch/models/cnnseg"

TRAIN_SIZE = 0.7
VALID_SIZE = 0.2
TEST_SIZE = 0.1

MODEL_NAME = "nvidia-speed"
# available model are: nvidia-speed, lstm

EPOCH_SEG = 1
EPOCH_SEG = 10

BATCH_SIZE = 8

LEARNING_RATE = 1e-3

LOSS_FUNC = 'mse'

OPTIMIZER = 'adam'
