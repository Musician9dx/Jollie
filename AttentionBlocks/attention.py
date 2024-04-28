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


class VAE_AttentionBlock(Layer):

    def __init__(self, numHeads, keyDim):
        super(VAE_AttentionBlock, self).__init__()

        self.normalization1 = LayerNormalization()
        self.normalization2 = LayerNormalization()

        self.mah = MultiHeadAttention(numHeads, keyDim, dropout=0.2)
        self.add = Add()

    def call(self, x, y, mask=False):
        residue = x
        x = self.normalization1(x)
        y = self.normalization2(y)
        x = self.mah(x, y)
        x = self.add([residue, x])

        return x

class Cross_Attention_Block(Layer):

    def __init__(self, numHeads):
        super(Cross_Attention_Block, self).__init__()
        self.numHeads = numHeads

        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.mah = MultiHeadAttention(self.numHeads, 8)

    def call(self, x, y):
        x = self.ln1(x)
        y = self.ln2(y)

        op = self.mah(x, y)

        return op

