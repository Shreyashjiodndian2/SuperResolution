from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, Reduction
from tensorflow import reduce_mean

class Losses:
    def __init__(self, numReplicas):
        self.numReplicas = numReplicas
    
    def bceLoss(self, real, pred):
        bce = BinaryCrossentropy(reduction=Reduction.NONE)
        loss = bce(real, pred)
        loss = reduce_mean(loss) * (1./self.numReplicas)
        return loss
    def mseLoss(self, real, pred):
        mse = MeanSquaredError(reduction=Reduction.NONE)
        loss = mse(real, pred)
        loss = reduce_mean(loss) * (1./self.numReplicas)
        return loss
    