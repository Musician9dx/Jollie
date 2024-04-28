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


class jollie(Model):

    def __init__(self, encoder, decoder, clip, time, prior, codebook):
        super(jollie, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.clipNLP = clip
        self.time = time
        self.prior = prior
        self.codeBook = codebook
        self.optimizer = Adam()

    def call(self, time, tokens, image):
        timeEmbeddings = self.time(time)

        imageEmbeddings = self.encoder(image)
        imageEmbeddings = self.codeBook(imageEmbeddings)

        textEmbeddings = self.clipNLP(timeEmbeddings)

        prior = self.prior(imageEmbeddings, textEmbeddings, timeEmbeddings)

        decodedImage = self.decoder(prior)

        return decodedImage

    def sampleData(self):

        # Data Base Connectors

    def lossFunction(selfself,true,pred):

        return tf.keras.losses.mean_squared_error(true,pred)

    def train(self):
        time, tokens, image, target = self.sampleData()

        for i in range(100):
            with tf.GradientTape() as tape:
                logits = self.call(time, tokens, image)
                loss = self.lossFunction(logits, target)
                grads = tape.gradient(loss, self.trainable_variables)

            print(f"Step {i}:", tf.reduce_mean(loss))

            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))