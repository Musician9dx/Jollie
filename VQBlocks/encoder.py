class VQCLIP_Encoder(Layer):

    def __init__(self):
        super(VQCLIP_Encoder, self).__init__()
        self.residual_block_1d = VAE_ResidualBlock(128, 128)
        self.residual_block_2d = VAE_ResidualBlock(128, 128)
        self.residual_block_3d = VAE_ResidualBlock(256, 256)
        self.residual_block_4d = VAE_ResidualBlock(256, 256)
        self.residual_block_5d = VAE_ResidualBlock(512, 512)
        self.residual_block_6d = VAE_ResidualBlock(512, 512)
        self.residual_block_7d = VAE_ResidualBlock(512, 512)
        self.residual_block_8d = VAE_ResidualBlock(512, 512)
        self.residual_block_9d = VAE_ResidualBlock(512, 512)
        self.residual_block_10d = VAE_ResidualBlock(512, 512)

        self.conv_2d_1d = Conv2D(128, kernel_size=3, strides=(1, 1), padding="same", activation="relu")
        self.conv_2d_2d = Conv2D(128, kernel_size=3, strides=(2, 2), padding="same", activation="relu")
        self.conv_2d_3d = Conv2D(256, kernel_size=3, strides=(2, 2), padding="same", activation="relu")
        self.conv_2d_4d = Conv2D(512, kernel_size=3, strides=(2, 2), padding="same", activation="relu")
        self.conv_2d_5d = Conv2D(128, kernel_size=3, strides=(2, 2), padding="same", activation="relu")
        self.conv_2d_6d = Conv2D(8, kernel_size=3, strides=(1, 1), padding="same", activation="relu")

        self.attention_block = VAE_AttentionBlock(8, 4)
        self.normalizationd = LayerNormalization()

    def call(self, x):
        x = self.conv_2d_1d(x)
        x = self.residual_block_1d(x)
        x = self.residual_block_2d(x)

        x = self.conv_2d_2d(x)

        x = self.residual_block_3d(x)
        x = self.residual_block_4d(x)

        x = self.conv_2d_3d(x)

        x = self.residual_block_5d(x)
        x = self.residual_block_6d(x)

        x = self.conv_2d_4d(x)

        x = self.residual_block_7d(x)
        x = self.residual_block_8d(x)
        x = self.residual_block_9d(x)

        x = self.conv_2d_5d(x)
        x = self.attention_block(x, x)
        x = self.residual_block_10d(x)

        x = self.normalizationd(x)
        x = self.conv_2d_6d(x)

        return x