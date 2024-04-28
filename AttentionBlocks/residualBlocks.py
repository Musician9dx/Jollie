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
from AttentionBlocks.attention import  VAE_AttentionBlock,Cross_Attention_Block



class VAE_ResidualBlock(Layer):
    def __init__(self, input_dim, output_dim):
        super(VAE_ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.normalization_1 = LayerNormalization()
        self.normalization_2 = LayerNormalization()
        self.conv_2d_1 = Conv2D(filters=self.input_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")
        self.conv_2d_2 = Conv2D(filters=self.output_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")
        self.conv_2d_3 = Conv2D(filters=self.output_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")

        self.add = Add()

    def call(self, x):

        input_dim = x.shape[-1]

        residue = x

        x = self.normalization_1(x)
        x = self.conv_2d_1(x)
        x = self.normalization_2(x)
        x = self.conv_2d_2(x)

        output_dim = x.shape[-1]

        if input_dim == output_dim:

            return x

        else:

            residue = self.conv_2d_3(residue)
            x = self.add([residue, x])
            return x

class Time_Embedding(Layer):

    def __init__(self, embedDim, outputDim, reshape_dim):
        super(Time_Embedding, self).__init__()

        self.embedDim = embedDim
        self.outputDim = outputDim

        self.embedding = Embedding(1, self.embedDim)

        self.flat = Flatten()
        self.linear_1 = Dense(self.outputDim, activation="linear")
        self.linear_2 = Dense(self.outputDim, activation="linear")

        self.linear_3 = Dense(self.outputDim, activation="linear")
        self.linear_4 = Dense(self.outputDim, activation="linear")

        self.reshape = Reshape(reshape_dim)

    def call(self, x):
        x = self.embedding(x)

        x = self.linear_1(x)
        x = self.linear_2(x)

        x = self.flat(x)

        x = self.linear_3(x)
        x = self.linear_4(x)

        x = self.reshape(x)

        return x