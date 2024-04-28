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
from ClipBlocks.cliplayers import CLIP_NLP
from VQBlocks.encoder import VQCLIP_Encoder


class CLIP(Model):

    def __init__(self):
        super(CLIP, self).__init__()
        self.nlpLayer = CLIP_NLP(seqLength=20, embedDim=8, numHead=8, outputDim=512, reshape_dim=(8, 8, 8))
        self.encoder = VQCLIP_Encoder()
        self.loss = CosineSimilarity()
        self.flat = Flatten()
        self.image = tf.random.uniform((1, 128, 128, 3))
        self.text = tf.constant([1, 2, 3])
        self.optimizer = Adam()

    def train(self, steps=10):
        for step in range(steps):
            with tf.GradientTape() as tape:
                image_embeddings = self.flat(self.encoder(self.image))
                text_embeddings = self.flat(self.nlpLayer(self.text))

                loss = self.loss(image_embeddings, text_embeddings)

                grads = tape.gradient(loss, self.trainable_variables)

            print(f"Step {step}: ", float(loss))

            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def returnModels(self):
        return self.nlpLayers, self.self.encoder

