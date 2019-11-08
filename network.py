from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, LeakyReLU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from layers import conv_block_2d, apply_padding


class VGG11():
    def __init__(self, config):
        self.config = config

    def define_model(self, model_name):
        img = Input(shape=self.config.data.img_shape, name=f"{model_name}_input")
        img_32 = apply_padding(pad_type="reflect", padding=[2, 2])(
            img)  # resizing the image by applying padding on tensor form
        x = conv_block_2d(img_32, 64, kernel_size=3, strides=1, pad_type="reflect", useBN=True, activation="lrelu")
        x = MaxPooling2D()(x)
        x = conv_block_2d(x, 128, kernel_size=3, strides=1, pad_type="reflect", useBN=True, activation="lrelu")
        x = MaxPooling2D()(x)
        x = conv_block_2d(x, 256, kernel_size=3, strides=1, pad_type="reflect", useBN=True, activation="lrelu")
        x = conv_block_2d(x, 256, kernel_size=3, strides=1, pad_type="reflect", useBN=True, activation="lrelu")
        x = MaxPooling2D()(x)
        x = conv_block_2d(x, 512, kernel_size=3, strides=1, pad_type="reflect", useBN=True, activation="lrelu")
        x = conv_block_2d(x, 512, kernel_size=3, strides=1, pad_type="reflect", useBN=True, activation="lrelu")
        x = MaxPooling2D()(x)
        x = conv_block_2d(x, 512, kernel_size=3, strides=1, pad_type="reflect", useBN=True, activation="lrelu")
        x = conv_block_2d(x, 512, kernel_size=3, strides=1, pad_type="reflect", useBN=True, activation="lrelu")
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(100)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(10)(x)
        scores = Activation("softmax")(x)
        return Model(img, scores, name=model_name)

    def compile_model(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = self.define_model("VGG11")
            opt = Adam(lr=self.config.model.lr)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
        return model
