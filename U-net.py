from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def unet():
    l2_lambda = 0.01
    input_shape = (64, 280, 1)
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(conv4)

    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(conv7)

    # Output
    output = Conv2D(1, 1, activation='relu')(conv7)

    model = Model(inputs=inputs, outputs=output)

    return model


# Create the U-Net model with L2 regularization

model = unet()
model.summary()
