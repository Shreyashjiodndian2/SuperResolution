from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model

class VGG:
    @staticmethod
    def build():
        vgg = VGG19(include_top=False, weights="imagenet", input_shape=(None, None, 3))
        # vgg.trainable = False
        model = Model(vgg.input,vgg.layers[20].output)
        return model
    