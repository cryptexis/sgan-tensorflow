import numpy as np
from data_util import save_tensor, get_data, save_tensor_rect
import tensorflow as tf


def visualize_generate(sess, sgan, reg=False):

    opt = tf.app.flags.FLAGS
    sgan.batch_size = opt.generate_samples
    sgan.z_dim = [opt.generate_height, opt.generate_width, opt.z_dim]
    print sgan.z_dim, sgan.batch_size

    sgan.build_model()
    sgan.load(opt.model_dir)
    z_sample = np.random.uniform(-1, 1, [sgan.batch_size] + sgan.z_dim).astype(np.float32)

    for i in range(0, sgan.batch_size):
        if reg:
            samples = sess.run(sgan.GD_net, feed_dict={sgan.random_noise: z_sample})
        else:
            samples = sess.run(sgan.G_net, feed_dict={sgan.random_noise: z_sample})
        save_tensor(samples[0], 'test_'+str(i)+'.jpg')