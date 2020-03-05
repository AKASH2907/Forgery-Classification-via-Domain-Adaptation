# import the necessary packages
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    Activation,
    BatchNormalization,
)


def vgg_16(inputs):
    print("VGG 16 model loaded ......")
    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3 & 4
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 5, 6, & 7
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x


def alexnet(inputs):
    print("AlexNet model loaded ......")
    x4 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="valid")(
        inputs
    )
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)
    x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x4)

    x4 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="valid")(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)
    x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x4)

    x4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)

    x4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)

    x4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)
    x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x4)

    return x4
