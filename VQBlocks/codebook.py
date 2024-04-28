class CodeBook(Layer):

    def __init__(self, numVectors, embedDim):

        self.numVectors = numVectors
        self.embedDim = embedDim

        super(CodeBook, self).__init__()
        self.CodeBookVectors = tf.Variable(tf.random.uniform((self.numVectors, self.embedDim)), trainable=True)

    def call(self, x):

        continous_vectors = x
        codebook_vectors = self.CodeBookVectors

        discreteImage = []

        Images = []

        for image in continous_vectors:

            discreteRow = []

            for row in image:

                discreteColumn = []

                for column in row:

                    errorVector = []

                    for j in codebook_vectors:
                        error = float((tf.keras.losses.mean_squared_error(column, j)))

                        errorVector.append(error)

                    discreteColumn.append(tf.argmin(errorVector))

                discreteRow.append(discreteColumn)

            Image = []

            for discreteColumn in discreteRow:

                ImageRow = []

                for argmin in discreteColumn:
                    ImageRow.append(codebook_vectors[argmin])

                Image.append(ImageRow)

            Images.append(Image)

        return tf.Variable(Images)
