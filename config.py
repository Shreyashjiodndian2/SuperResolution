import os

DATASET = "div2k/bicubic_x4"

SHARD_SIZE = 256
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 8

HR_SHAPE = [96, 96,3]
LR_SHAPE = [24, 24,3]
SCALING_FACTOR = 4

FEATURE_MAPS = 64
RESIDUAL_BLOCKS = 16
LEAKY_ALPHA = 0.2
DISC_BLOCKS = 4

PRETRAIN_LR = 1e-4
FINETUNE_LR = 1e-5
PRETRAIN_EPOCHS = 2500
FINETUNE_EPOCHS = 2500
STEPS_PER_EPOCH = 10

BASE_DATA_PATH = "dataset"
DIV2K_PATH = os.path.join(BASE_DATA_PATH, "DIV2K".lower())

GPU_BASE_TFR_PATH = "tfrecord"
GPU_DIV2K_TFR_TRAIN_PATH = os.path.join(GPU_BASE_TFR_PATH, "train")
GPU_DIV2K_TFR_TEST_PATH = os.path.join(GPU_BASE_TFR_PATH, "test")

BASE_OUTPUT_PATH = "outputs"

GPU_PRETRAINED_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH,
                                              "models", "pretrained_generator")
GPU_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models",
                                   "generator")

BASE_IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, "images")
GRID_IMAGE_PATH = os.path.join(BASE_IMAGE_PATH, "grid.png")

