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
from AttentionBlocks.attention import VAE_AttentionBlock,Cross_Attention_Block

class Prior(Layer):

    def __init__(self):

        super(Prior, self).__init__()

        self.timeTextAttention = Cross_Attention_Block(8)
        self.textImageAttention = Cross_Attention_Block(8)
        self.selfAttentionBlocks = [VAE_AttentionBlock(8, 16) for i in range(12)]
        self.denseLayers = [Dense(8, activation="linear") for i in range(8)]

    def call(self, image, text, time):

        timeText = self.timeTextAttention(text, time)
        textImage = self.textImageAttention(image, timeText)

        x = textImage

        for layer in self.selfAttentionBlocks:
            x = layer(x, x)

        for layer in self.denseLayers:
            x = layer(x)

        return x