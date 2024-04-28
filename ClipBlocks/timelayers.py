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