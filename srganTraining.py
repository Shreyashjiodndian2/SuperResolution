from tensorflow.keras import Model
from tensorflow import GradientTape, concat, zeros, ones
import tensorflow as tf

class SRGANTraining(Model):
    def __init__(self, generator, discriminator, vgg, batchSize):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.batchSize = batchSize
    
    def compile(self, generatorOptimizer, discriminatorOptimizer, bceLoss, mseLoss):
        super().compile()
        self.generatorOptimizer = generatorOptimizer
        self.discriminatorOptimizer = discriminatorOptimizer
        self.bceLoss = bceLoss
        self.mseLoss = mseLoss
    def trainStep(self, images):
        (lrImages, hrImages) = images
        lrImages = tf.cast(lrImages, tf.float32)
        hrImages = tf.cast(hrImages, tf.float32)
        
        srImages = self.generator(lrImages, training=True)
        combinedImages = concat([lrImages, srImages], axis=0)
        
        label = concat([zeros((self.batchSize,1)), ones((self.batchSize, 1))], axis=0)
        with GradientTape() as tape:
            predictions = self.discriminator(combinedImages, training=True)
            discriminatorLoss = self.bceLoss(label, predictions)
        grads = tape.gradient(discriminatorLoss, self.discriminator.trainable_variables)
        self.discriminatorOptimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        misLeadingLabels = ones((self.batchSize, 1))
        with GradientTape() as tape:
            srImages = self.generator(lrImages, training=True)
            predictions = self.discriminator(srImages, training=True)
            generatorLoss =1e-3* self.bceLoss(misLeadingLabels, predictions)
            srVgg = tf.keras.applications.vgg19.preprocess_input(srImages)
            srVgg = self.vgg(srVgg)/12.75
            hrVgg = tf.keras.applications.vgg19.preprocess_input(hrImages)
            hrVgg = self.vgg(hrVgg)/12.75
            
            percLoss = self.mseLoss(hrVgg, srVgg)
            generatorTotalLoss = generatorLoss + percLoss
        grads = tape.gradient(generatorTotalLoss, self.generator.trainable_variables)
        self.generatorOptimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return {"discriminatorLoss":discriminatorLoss, "generatorTotalLoss":generatorTotalLoss, "generatorLoss":generatorLoss, "percLoss":percLoss}