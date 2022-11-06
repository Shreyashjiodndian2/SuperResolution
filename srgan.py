from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Conv2D, LeakyReLU, Add, PReLU, Rescaling
from tensorflow.nn import depth_to_space
from tensorflow.keras import Model, Input


class SRGAN(object):
    @staticmethod
    def generator(scalingFactor, featureMaps, residualBlocks):
        inputs = Input(shape=(None, None, 3))
        xIn = Rescaling(scale=1./255., offset=0)(inputs)
        xIn = Conv2D(featureMaps, 9, padding="same")(xIn)
        xIn = PReLU(shared_axes=[1, 2])(xIn)
        x = Conv2D(featureMaps, 3, padding="same")(xIn)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        xSkip = Add()([xIn, x])
        for _ in range(residualBlocks-1):
            x = Conv2D(featureMaps, 3, padding="same")(xSkip)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(featureMaps, 3, padding="same")(x)
            x = BatchNormalization()(x)
            xSkip = Add()([xSkip, x])
        x = Conv2D(featureMaps, 3, padding="same")(xSkip)
        x = BatchNormalization()(x)
        x = Add()([xIn, x])
        x = Conv2D(featureMaps * (scalingFactor // 2), 3, padding="same")(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2])(x)

        x = Conv2D(featureMaps * scalingFactor, 3, padding="same")(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2])(x)

        x = Conv2D(3, 9, padding="same", activation="tanh")(x)
        x = Rescaling(scale=127.5, offset=127.5)(x)
        generator = Model(inputs, x, name="generator")
        return generator

    @staticmethod
    def discriminator(featureMaps, leakyAlpha, discBlocks):
        inputs = Input(shape=(None, None, 3))
        x = Rescaling(scale=1./127.5, offset=-1.0)(inputs)
        x = Conv2D(featureMaps, 3, padding="same")(x)

        x = LeakyReLU(leakyAlpha)(x)

        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leakyAlpha)(x)

        for i in range(1, discBlocks):
            x = Conv2D(featureMaps * (2 ** i), 3, strides=2, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)

            x = Conv2D(featureMaps * (2 ** i), 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)
        x = GlobalAveragePooling2D()(x)
        x = LeakyReLU(leakyAlpha)(x)

        x = Dense(1, activation="sigmoid")(x)

        discriminator = Model(inputs, x, name="discriminator")
        return discriminator
