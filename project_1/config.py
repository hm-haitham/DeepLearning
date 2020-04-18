import torch.nn as nn

NB_SAMPLES = 1000
DATA_DIR = './data'

WIDTH_HEIGHT = 14
SAMPLE_SIZE = 392

# ----Train Config-----#
LEARNING_RATE = 0.00001
TRAIN_BATCH_SIZE = 1
CRITERION = nn.BCELoss()
EPOCHS = 10

TRAIN_CHECKPOINTS_DIR = "./checkpoints"
SAVE_MODEL_EVERY_X_EPOCH = 5

#----Test Config-----#
TEST_BATCH_SIZE = NB_SAMPLES

#----BasicNet Config-----#
BASIC_NET_NAME = "basic_net"
BASIC_NET_HIDDEN_LAYER = 2048
