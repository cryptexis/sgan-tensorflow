import tensorflow as tf


class BatchNorm(object):

    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def convolution(input_tensor, output_dim, stddev=0.02, name="conv2d"):
    """

    :param input_tensor:
    :param output_dim:
    :param kernel_size:
    :param stride:
    :param stddev:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        kernel_size = tf.app.flags.FLAGS.kernel_size
        shape = [kernel_size, kernel_size, input_tensor.get_shape()[-1], output_dim]
        w = tf.get_variable('W',
                            shape,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        convolve = tf.nn.conv2d(input_tensor,
                                w,
                                strides=[1, 2, 2, 1],
                                padding='SAME')

        return convolve, w


def deconvolution(input_tensor, output_shape, suffix, stddev=0.02, name="deconv2d"):

    kernel_size = tf.app.flags.FLAGS.kernel_size

    with tf.variable_scope(name):
        shape = [kernel_size, kernel_size, output_shape[-1], input_tensor.get_shape()[-1]]
        w = tf.get_variable('W_'+str(suffix),
                            shape,
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconvolve = tf.nn.conv2d_transpose(input_tensor,
                                                w,
                                                output_shape=output_shape,
                                                strides=[1, 2, 2, 1],
                                                padding='SAME')
        except AttributeError:
            deconvolve = tf.nn.conv2d_transpose(input_tensor,
                                        w,
                                        output_shape=output_shape,
                                        strides=[1, 2, 2, 1])

        return deconvolve, w

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)