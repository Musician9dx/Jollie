import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, Multiply, Conv2D, Add, Concatenate, MaxPooling2D, LayerNormalization, Layer, UpSampling2D, Reshape, Flatten, Conv2DTranspose
from tensorflow.keras import Model
from keras_nlp.layers import TokenAndPositionEmbedding, StartEndPacker
from tensorflow import GradientTape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow import GradientTape
from einops import repeat,rearrange
from tensorflow.keras.losses import CosineSimilarity
from VQBlocks import encoder,decoder,codebook


class VQGAN(Model):

    def __init__(self):
        super(VQGAN, self).__init__()
        self.encoder = VQCLIP_Encoder()
        self.codebook = CodeBook(100, 8)
        self.decoder = VQ_Decoder()
        self.imageGenClassifier = """ImageGenClassifier()"""

    def call(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return x.shape, y.shape

    def sample_data(self):
        # Data Base Connectors
        pass

    def train(self, steps):
        for i in range(steps):
            inputImage, target = sample_data()

            with tf.GradientTape() as tape:
                logits = self.encoder(inputImage)
                logits = self.codebook(logits)
                logits = self.decoder(logits)

                loss = self.imageGenClassifier(logits)

                grads = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.non_trainable_variables))

    def returnModels(self):
        return self.encoder, self.decoder, self.codebook