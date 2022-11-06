import sys
import os
import argparse
from tensorflow.io.gfile import glob
from tensorflow.keras.optimizers import Adam
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.config import experimental_connect_to_cluster
from tensorflow import distribute
from losses import Losses
import config
from srganTraining import SRGANTraining
from vgg import VGG
from srgan import SRGAN
from dataPreProcess import loadDataset
import tensorflow as tf
tf.random.set_seed(42)


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--device", type=str, default="gpu",
                choices=["gpu", "tpu", "cpu"], help="device to train on")
args = vars(ap.parse_args())

if args["device"] == "tpu":
    tpu = distribute.cluster_resolver.TPUClusterResolver()
    experimental_connect_to_cluster(tpu)
    initialize_tpu_system(tpu)
    strategy = distribute.TPUStrategy(tpu)
elif args["device"] == "gpu":
    strategy = distribute.MirroredStrategy()
    tfrTrainPath = config.GPU_DIV2K_TFR_TRAIN_PATH
    preTrainGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
    generatorPath = config.GPU_GENERATOR_MODEL
else:
    print("Training on CPU not possible")
    sys.exit(0)

print("Number of devices: {}".format(strategy.num_replicas_in_sync))
print("Grabbing the train TFRecords")
trainTfr = glob(tfrTrainPath+"/*.tfrec")

print("Creating train and test datasets")
trainDs = loadDataset(trainTfr, config.TRAIN_BATCH_SIZE *
                      strategy.num_replicas_in_sync)
with strategy.scope():
    losses = Losses(strategy.num_replicas_in_sync)
    generator = SRGAN().generator(
        scalingFactor=config.SCALING_FACTOR,
        featureMaps=config.FEATURE_MAPS,
        residualBlocks=config.RESIDUAL_BLOCKS
    )
    generator.compile(
        optimizer=Adam(learning_rate=config.PRETRAIN_LR),
        loss=losses.mseLoss
    )
    print("Loading the pre-trained generator")
    print(trainDs)
    generator.fit(trainDs, epochs=config.PRETRAIN_EPOCHS,
                  steps_per_epoch=config.STEPS_PER_EPOCH)

if args["device"] == "gpu" and not os.path.exists(config.BASE_OUTPUT_PATH):
    os.makedirs(config.BASE_OUTPUT_PATH)

print("Saving the pre-trained generator to {}".format(preTrainGenPath))
generator.save(preTrainGenPath)

with strategy.scope():
    losses = Losses(strategy.num_replicas_in_sync)
    vgg = VGG().build()
    discriminator = SRGAN().discriminator(
        featureMaps=config.FEATURE_MAPS,
        leakyAlpha=config.LEAKY_ALPHA, discBlocks=config.DISC_BLOCKS)
    srgan = SRGANTraining(generator, discriminator, vgg,
                          batchSize=config.TRAIN_BATCH_SIZE)
    srgan.compile(
        discriminatorOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        generatorOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        bceLoss=losses.bceLoss,
        mseLoss=losses.mseLoss)
    print("Training the SRGAN")
    srgan.fit(trainDs, epochs=config.FINETUNE_EPOCHS,
              steps_per_epoch=config.STEPS_PER_EPOCH)

print("Saving the generator to {}".format(generatorPath))
srgan.generator.save(generatorPath)
