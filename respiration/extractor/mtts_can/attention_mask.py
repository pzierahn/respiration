import keras
import tensorflow as tf


class AttentionMask(keras.layers.Layer):
    def call(self, x):
        xsum = tf.reduce_sum(x, axis=1, keepdims=True)
        xsum = tf.reduce_sum(xsum, axis=2, keepdims=True)
        return x / xsum * x.shape[1] * x.shape[2] * 0.5

    def get_config(self):
        config = super(AttentionMask, self).get_config()
        return config
