import tensorflow as tf
from keras.layers import Conv2D


class TSM(tf.keras.layers.Layer):
    """
    Temporal Shift Module
    """
    n_frame: int
    fold_div: int

    def __init__(self, n_frame, fold_div=3, **kwargs):
        super(TSM, self).__init__(**kwargs)
        self.n_frame = n_frame
        self.fold_div = fold_div

    def call(self, x):
        nt, h, w, c = x.shape
        x = tf.reshape(x, (-1, self.n_frame, h, w, c))
        fold = c // self.fold_div
        last_fold = c - (self.fold_div - 1) * fold
        out1, out2, out3 = tf.split(x, [fold, fold, last_fold], axis=-1)

        # Shift left
        padding_1 = tf.zeros_like(out1)
        padding_1 = padding_1[:, -1, :, :, :]
        padding_1 = tf.expand_dims(padding_1, 1)
        _, out1 = tf.split(out1, [1, self.n_frame - 1], axis=1)
        out1 = tf.concat([out1, padding_1], axis=1)

        # Shift right
        padding_2 = tf.zeros_like(out2)
        padding_2 = padding_2[:, 0, :, :, :]
        padding_2 = tf.expand_dims(padding_2, 1)
        out2, _ = tf.split(out2, [self.n_frame - 1, 1], axis=1)
        out2 = tf.concat([padding_2, out2], axis=1)

        out = tf.concat([out1, out2, out3], axis=-1)
        out = tf.reshape(out, (-1, h, w, c))

        return out

    def get_config(self):
        config = super(TSM, self).get_config()
        return config


def tsm_cov_2d(x, n_frame, nb_filters=128, kernel_size=(3, 3), activation='tanh', padding='same'):
    x = TSM(n_frame=n_frame)(x)
    x = Conv2D(nb_filters, kernel_size, padding=padding, activation=activation)(x)
    return x
