from tensorflow.io import FixedLenFeature, parse_single_example, parse_tensor
from tensorflow.image import flip_left_right, rot90
import tensorflow as tf

AUTO = tf.data.AUTOTUNE


def randomCrop(lrImage, hrImage, hrCropSize=96, scale=4):
    lrCropSize = hrCropSize // scale
    lrImageShape = tf.shape(lrImage)[:2]

    lrW = tf.random.uniform(
        shape=(), maxval=lrImageShape[1] - lrCropSize + 1, dtype=tf.int32)
    lrH = tf.random.uniform(
        shape=(), maxval=lrImageShape[0] - lrCropSize + 1, dtype=tf.int32)

    hrW = lrW * scale
    hrH = lrH * scale

    lrImageCropped = tf.slice(lrImage, [lrH, lrW, 0], [
                              lrCropSize, lrCropSize, 3])
    hrImageCropped = tf.slice(hrImage, [hrH, hrW, 0], [
                              hrCropSize, hrCropSize, 3])

    return (lrImageCropped, hrImageCropped)


def getCenterCrop(lrImage, hrImage, hrCropSize=96, scale=4):
    lrCropSize = hrCropSize // scale
    lrImageShape = tf.shape(lrImage)[:2]

    lrW = (lrImageShape[1]) // 2
    lrH = (lrImageShape[0]) // 2

    hrW = lrW * scale
    hrH = lrH * scale

    lrImageCropped = tf.slice(lrImage, [lrH, lrW, 0], [
                              lrCropSize, lrCropSize, 3])
    hrImageCropped = tf.slice(hrImage, [hrH, hrW, 0], [
                              hrCropSize, hrCropSize, 3])

    return (lrImageCropped, hrImageCropped)


def randomFlip(lrImage, hrImage):
    doFlip = tf.random.uniform(shape=(), maxval=1.0)
    if doFlip > 0.5:
        lrImage = flip_left_right(lrImage)
        hrImage = flip_left_right(hrImage)
    return (lrImage, hrImage)


def randomRotate(lrImage, hrImage):
    numRotations = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    lrImage = rot90(lrImage, k=numRotations)
    hrImage = rot90(hrImage, k=numRotations)
    return (lrImage, hrImage)


def readTrainExample(example):
    features = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string)
    }
    example = parse_single_example(example, features)
    lrImage = parse_tensor(example["lr"], out_type=tf.float32)
    hrImage = parse_tensor(example["hr"], out_type=tf.float32)
    lrImage, hrImage = randomCrop(lrImage, hrImage)
    lrImage, hrImage = randomFlip(lrImage, hrImage)
    lrImage, hrImage = randomRotate(lrImage, hrImage)
    return (lrImage, hrImage)


def readTestExample(example):
    features = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string)
    }
    example = parse_single_example(example, features)
    lrImage = parse_tensor(example["lr"], out_type=tf.float32)
    hrImage = parse_tensor(example["hr"], out_type=tf.float32)
    lrImage, hrImage = getCenterCrop(lrImage, hrImage)
    lrImage = tf.reshape(lrImage, [24, 24, 3])
    hrImage = tf.reshape(hrImage, [96, 96, 3])
    return (lrImage, hrImage)


def loadDataset(fileNames, batchSize, train=False):
    dataset = tf.data.TFRecordDataset(fileNames, num_parallel_reads=AUTO)
    if train:
        dataset = dataset.map(readTrainExample, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(readTestExample, num_parallel_calls=AUTO)
    dataset = (dataset.shuffle(batchSize).batch(
        batchSize).repeat().prefetch(AUTO))
    return dataset
