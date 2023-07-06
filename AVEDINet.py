from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from ERINN.erinn.metrics import r_squared
from tensorflow.python.ops.image_ops_impl import total_variation

class VEDI():
    def __init__(self):
        self.input_size1 = (64, 280, 1)
        self.input_size2 = (64, 280, 1)
        inputs1 = Input(self.input_size1)
        inputs2 = Input(self.input_size2)
        self.encoder1 = self.encoder(inputs1)
        self.encoder2 = self.encoder(inputs2)
        self.attention = self.attention()
        merged = self.attention([self.encoder1.output, self.encoder2.output])
        self.decoder = self.decoder()
        outputs = self.decoder(merged)
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.vae_loss, optimizer=opt, metrics=[r_squared])
        self.input_shape1 = self.model.input_shape[0]
        self.input_shape2 = self.model.input_shape[1]
        self.output_shape = self.model.output_shape

    def residual_block(self, x, output_filters, stride):
        res_x = Conv2D(kernel_size=(3, 3), filters=output_filters, strides=stride, padding='same',
                       kernel_regularizer=l2(0.01))(x)
        res_x = BatchNormalization()(res_x)
        res_x = Activation('relu')(res_x)
        res_x = Conv2D(kernel_size=(3, 3), filters=output_filters, strides=1, padding='same',
                       kernel_regularizer=l2(0.01))(res_x)
        res_x = BatchNormalization()(res_x)
        identity = Conv2D(kernel_size=(1, 1), filters=output_filters, strides=stride, padding='same')(x)
        x = Add()([identity, res_x])
        output = Activation('relu')(x)
        return output

    def encoder(self, inputs):
        x = self.residual_block(inputs, 16, 2)
        x = self.residual_block(x, 32, 2)
        x = self.residual_block(x, 64, 2)
        x = self.residual_block(x, 128, 2)
        x = self.residual_block(x, 256, 2)
        x = self.residual_block(x, 512, 2)
        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        z_mean = Dense(512, activation='relu', kernel_initializer='he_normal',
                            kernel_regularizer=l2(0.01), name='z_mean')(x)
        z_log_var = Dense(512, activation='relu', kernel_initializer='he_normal',
                               kernel_regularizer=l2(0.01), name='z_log_var')(x)
        z = Lambda(self.sampling, output_shape=(512,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        return encoder

    def attention(self):
        def attention(inputs):
            x1, x2 = inputs
            attention_weights = Dot(axes=[1, 2])([x1,x2])
            attention_weights = Activation('softmax')(attention_weights)
            merged = Add()([x1, attention_weights * x2])
            return merged
        return attention

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def decoder(self):
        latent_inputs = Input(shape=(512,), name='z_sampling')
        x = Dense(1 * 4 * 512, activation='relu', kernel_regularizer=l2(0.01))(latent_inputs)
        x = Reshape((1, 4, 512))(x)
        x = Conv2DTranspose(1024, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, kernel_size=(2,3), strides=2, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, kernel_size=(2,3), strides=2, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, kernel_size=3, strides=(2,2), padding='same')(x)
        x = Activation('relu')(x)
        output = Conv2D(kernel_size=(3, 3), filters=1, strides=1, padding='same')(x)
        decoder = Model(inputs=latent_inputs, outputs=output, name='decoder')
        decoder.summary()
        return decoder

    def vae_loss(self, inputs, outputs):
        mse_loss = K.mean(K.square(inputs - outputs))
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.mean(kl_loss)
        kl_loss *= -0.5
        tv_loss = K.mean(total_variation(outputs))
        loss = 0.01 * kl_loss + mse_loss
        return loss

# 实例化 VEDI 类
vedi = VEDI()

# 打印模型结构
vedi.model.summary()
