'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''

BASE = "../ai-autoencoder-pack-similarities/autoencoder/"
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_TRAIN_PATH =  BASE +"data/datasets/train"
IMG_VAL_PATH = BASE +"data/datasets/val"
IMG_TEST_PATH =  BASE +"data/datasets/test"
ENC_MODEL_PATH = BASE +"outputs/weights/"
DEC_MODEL_PATH = BASE + "outputs/weights/"
MODEL_NAME = "baseline_encoder.pt"
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100
EMBEDDING_DIM = (1,64,64,64)
