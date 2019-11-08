import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, BatchNormalization, ReLU, Activation, LeakyReLU


def apply_padding(pad_type, padding):
    pad_w, pad_h = padding
    if pad_type == "reflect":
        return Lambda(lambda _x: tf.pad(_x, [[0, 0], [pad_w, pad_w], [pad_h, pad_h], [0, 0]], mode="REFLECT"))
    elif pad_type == "zero":
        return Lambda(lambda _x: tf.pad(_x, [[0, 0], [pad_w, pad_w], [pad_h, pad_h], [0, 0]], mode="CONSTANT"))

    else:
        raise ValueError(f"Unsupported pad type: {pad_type}")


def apply_activation(activation):
    if activation is None:
        return lambda x: x
    elif activation == "relu":
        return ReLU()
    elif activation == "lrelu":
        return LeakyReLU(0.2)
    elif activation == "tanh":
        return Activation("tanh")
    elif activation == "sigmoid":
        return Activation("sigmoid")
    else:
        raise ValueError(f"Unsuported activation: {activation}")


def conv_block_2d(x, filters, kernel_size=3, strides=1, pad_type="zero", useBN=False, activation=None,
                  **kwargs):
    # Padding
    pading = (kernel_size - 1) // 2
    x = apply_padding(pad_type, [pading, pading])(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="valid")(x)

    # Normalization
    if useBN:
        x = BatchNormalization()(x)

    # Activation
    x = apply_activation(activation)(x)
    return x
