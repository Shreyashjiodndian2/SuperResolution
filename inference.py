from dataPreProcess import loadDataset
from utils import zoomIntoImages
from . import config
from tensorflow import distribute
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.io.gfile import glob
from matplotlib.pyplot import subplots
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--device", type=str, default="gpu",
                choices=["gpu", "cpu"], help="device to train on")
args = vars(ap.parse_args())
if args["device"] == "gpu":
    strategy = distribute.MirroredStrategy()
    tfrTestPath = config.GPU_DIV2K_TFR_TEST_PATH
    preTrainedGeneratorPath = config.GPU_PRETRAINED_GENERATOR_MODEL
    generatorPath = config.GPU_GENERATOR_MODEL
else:
    print("Training on CPU not possible")
    sys.exit(0)

print("Loading the test dataset")
testTfr = glob(tfrTestPath+"/*.tfrec")
testDataset = loadDataset(testTfr, config.INFER_BATCH_SIZE, train=False)

(lrImage, hrImage) = next(iter(testDataset))

with strategy.scope():
    print("Loading the pre-trained generator and fully trained SRGAN")
    srganPreTrainedGenerator = load_model(
        preTrainedGeneratorPath, compile=False)
    srganGenerator = load_model(generatorPath, compile=False)

    srganPreTrainedGeneratorPrediction = srganPreTrainedGenerator.predict(
        lrImage)
    srganGeneratorPrediction = srganGenerator.predict(lrImage)

print("plotting the SRGAN predictions")
(fig, axes) = subplots(nrows=config.INFER_BATCH_SIZE, ncols=4, figsize=(50, 50))

for (ax, lowRes, srPreIm, srGanIm, highRes) in zip(axes, lrImage, srganPreTrainedGeneratorPrediction, srganGeneratorPrediction, hrImage):
    ax[0].imshow(array_to_img(lowRes))
    ax[0].set_title("Low Resolution")
    ax[1].imshow(array_to_img(srPreIm))
    ax[1].set_title("SRGAN Pre-trained Generator")
    ax[2].imshow(array_to_img(srGanIm))
    ax[2].set_title("SRGAN Generator")
    ax[3].imshow(array_to_img(highRes))
    ax[3].set_title("High Resolution")

if not os.path.exists(config.BASE_IMAGE_PATH):
    os.makedirs(config.BASE_IMAGE_PATH)

print("Saving the images")
fig.savefig(config.GRID_IMAGE_PATH)

zoomIntoImages(
    srganPreTrainedGeneratorPrediction[0], "SRGAN Pre-trained Generator")
zoomIntoImages(srganGeneratorPrediction[0], "SRGAN Generator")
