import tensorflow as tf
import numpy as np

class Mish(tf.keras.layers.Layer):
        def __init__(self):
            super(Mish, self).__init__()

        def call(self, inputs):
            return inputs * tf.math.tanh(tf.math.softplus(inputs))