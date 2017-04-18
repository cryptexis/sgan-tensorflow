import os
import numpy as np
import tensorflow as tf
from utils.tf_utils import visualize_generate

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate of for adam [0.0005]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("l2_decay", 1e-5, "L2 regularization weight")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("z_size", 9, "Spatial size of the random noise")
flags.DEFINE_integer("z_dim", 100, "Dimensionality of each vector")
flags.DEFINE_integer("max_iterations", 25001, "Dimensionality of each vector")
flags.DEFINE_integer("kernel_size", 5, "Convolutional Kernel Size in the network")
flags.DEFINE_integer("checkpoint_interval", 15, "Convolutional Kernel Size in the network")
flags.DEFINE_integer("num_layers", 5, "Convolutional Kernel Size in the network")
flags.DEFINE_string("dataset", "./texture", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("model_dir", "./checkpoint_reg/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("reg", False, "In order to turn on regularization")

# test phase
flags.DEFINE_integer("generate_samples", 1, "Numbers of samples to generate")
flags.DEFINE_integer("generate_width", 36, "Numbers of samples to generate")
flags.DEFINE_integer("generate_height", 36, "Numbers of samples to generate")

FLAGS = flags.FLAGS


if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

if FLAGS.reg:
    from sgan_reg import SpatialGan
else:
    from sgan import SpatialGan

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    sgan = SpatialGan(sess)

    if FLAGS.is_train:
        sgan.build_model()
        sgan.train()
    else:
        visualize_generate(sess, sgan, FLAGS.reg)
