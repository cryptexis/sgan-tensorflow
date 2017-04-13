import os
import time
import numpy as np
import re
from utils.data_util import get_data, save_tensor
from model.ops import *


class SpatialGan(object):

    real_images = None
    random_noise = None
    G_net = None
    G_loss = None
    G_vars = None
    D_net_real = None
    D_net_fake = None
    D_logits_real = None
    D_logits_fake = None
    D_loss_real = None
    D_loss_fake = None
    D_loss = None
    D_vars = None

    lr_sum = None
    lf_sum = None
    lg_sum = None
    lt_sum = None
    img_sum = None

    hyper_params = None
    saver = None

    def __init__(self, sess):

        self.opt = tf.app.flags.FLAGS
        self.sess = sess

        self.batch_size = self.opt.batch_size
        self.image_size = (2**self.opt.num_layers)*(self.opt.z_size - 1) + 1

        # define the dimensionality of the input tensor for discriminator network
        self.image_shape = [self.image_size, self.image_size, 3]
        # define the dimensionality of the input tensor for generator network
        self.z_dim = [self.opt.z_size, self.opt.z_size, self.opt.z_dim]

        # depending on the number of layers define the depth of each layer D_network
        self.d_layers = []
        self.d_filters = [2**(n+6) for n in range(self.opt.num_layers-1)]+[1]
        self.d_weights = []
        # same for the G_network
        self.g_layers = []
        self.g_weights = []
        self.g_filters = [3] + [2 ** (n + 6) for n in range(self.opt.num_layers - 1)]
        self.g_filters = self.g_filters[::-1]

        self.d_bn_layers = []
        self.g_bn_layers = []

        # batch normalization layers
        for i in range(0, self.opt.num_layers - 2):
            self.d_bn_layers.append(BatchNorm(name='d_bn'+str(i+1)))

        for i in range(0, self.opt.num_layers - 1):
            self.g_bn_layers.append(BatchNorm(name='g_bn'+str(i+1)))


    def build_model(self):
        """
        Model and Loss definitions
        :return: 
        """

        self.real_images = tf.placeholder(tf.float32,
                                          [self.batch_size] + self.image_shape,
                                          name='real_images')

        self.random_noise = tf.placeholder(tf.float32,
                                           [self.batch_size] + self.z_dim,
                                           name='z')
        # forward pass through generator
        self.G_net = self.generator(self.random_noise)
        # forward pass through discriminator / real images
        self.D_net_real, self.D_logits_real = self.discriminator(self.real_images)
        # forward pass through discriminator / generated images + reuse variables from the first network
        self.D_net_fake, self.D_logits_fake = self.discriminator(self.G_net, reuse=True)

        if self.opt.is_train:
            # discriminative loss on data points sampled from data distribution
            self.D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_real,
                                                        labels=tf.ones_like(self.D_logits_real)))
            self.lr_sum = tf.summary.scalar("D_loss_real", self.D_loss_real)
            # discriminative loss on data points sampled from generated distribution
            self.D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake,
                                                        labels=tf.zeros_like(self.D_logits_fake)))

            self.lf_sum = tf.summary.scalar("D_loss_fake", self.D_loss_fake)
            # generative loss
            self.G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake,
                                                        labels=tf.ones_like(self.D_logits_fake)))

            self.lg_sum = tf.summary.scalar("G_loss", self.G_loss)
            self.img_sum = tf.summary.image("output", self.G_net, 3)

            # Discriminative loss + L2 decay on weights
            self.D_loss = self.D_loss_real + self.D_loss_fake

            self.lt_sum = tf.summary.scalar("D_loss", self.D_loss)

            t_vars = tf.trainable_variables()

            self.D_vars = [var for var in t_vars if 'd_' in var.name]
            self.G_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(reshape=True, max_to_keep=self.opt.max_iterations)

    def train(self):
        """
        Training steps for Spatial GAN
        :return: 
        """

        # define optimizer for both networks
        d_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1=self.opt.beta1) \
            .minimize(self.D_loss, var_list=self.D_vars)

        g_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1=self.opt.beta1) \
            .minimize(self.G_loss, var_list=self.G_vars)

        tf.global_variables_initializer().run()

        g_sum = tf.summary.merge([self.lg_sum, self.img_sum])

        d_sum = tf.summary.merge([self.lr_sum, self.lf_sum, self.lt_sum])

        sample_image = np.random.uniform(-1, 1, [self.opt.batch_size]+self.z_dim).astype(np.float32)
        writer = tf.summary.FileWriter(self.opt.checkpoint_dir, self.sess.graph)

        start_time = time.time()
        epoch = 1

        while epoch < self.opt.max_iterations:

            batch_images = get_data(self.opt.dataset, self.image_size, self.batch_size)
            batch_z = np.random.uniform(-1, 1, [self.opt.batch_size]+self.z_dim).astype(np.float32)

            if epoch % 2 != 0:
                print "Update D:"
                # Update D network
                _, summary_str = self.sess.run([d_optim, d_sum],
                                               feed_dict={
                                                   self.real_images: batch_images,
                                                   self.random_noise: batch_z
                                                   })
            else:
                print "Update G:"
                # Update G network
                _, summary_str = self.sess.run([g_optim, g_sum],
                                               feed_dict={
                                                           self.random_noise: batch_z
                                                       })

            if epoch % self.opt.checkpoint_interval == 0:

                writer.add_summary(summary_str, epoch)
                img_out = self.G_net.eval({self.random_noise: sample_image})
                save_tensor(img_out[0], os.path.join(self.opt.sample_dir, str(epoch)+'.jpg'))

            if epoch % 100 == 0:
                self.saver.save(self.sess, os.path.join(self.opt.checkpoint_dir, 'fck.ckpt'), global_step=epoch)

            time_delta = time.time() - start_time
            print("Epoch: [%2d] time: %4.4f" % (epoch, time_delta))

            epoch += 1

    def discriminator(self, image, reuse=False):
        """
        In discriminator network batch normalization
        used on all layers except input and output
        :param image:
        :param reuse:
        :return:
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            # down sample the first layer
            conv, weight = convolution(image, self.d_filters[0], name='d_h0_conv')

            if not reuse:
                self.d_weights.append(weight)
            self.d_layers.append(lrelu(conv))

            for i in range(0, self.opt.num_layers - 2):
                conv, weight = convolution(self.d_layers[-1],
                                           self.d_filters[i+1],
                                           name='d_h'+str(i+1)+'_conv')
                if not reuse:
                    self.d_weights.append(weight)
                self.d_layers.append(
                    lrelu(
                        self.d_bn_layers[i](conv)
                    )
                )
            # last layer
            logit, weight = convolution(self.d_layers[-1], self.d_filters[-1], name='d_h4_conv')
            if not reuse:
                self.d_weights.append(weight)
            self.d_layers.append(tf.nn.sigmoid(logit))

        return self.d_layers[-1], logit

    def generator(self, z, reuse=False, train=True):
        """
        In generator network batch normalization used
        to all layers except the output layer
        :param z:
        :param reuse:
        :param train:
        :return:
        """
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            if train:
                _, h, w, in_channels = [i.value for i in z.get_shape()]
            else:
                sh = tf.shape(z)
                h = tf.cast(sh[1], tf.int32)
                w = tf.cast(sh[2], tf.int32)

            self.g_layers.append(z)
            # upscale image num_layers times
            for i in range(0, self.opt.num_layers - 1):
                new_h = (2**(i+1))*(h-1)+1
                new_w = (2**(i+1))*(w-1)+1
                out_shape = [self.batch_size, new_h, new_w, self.g_filters[i]]

                # deconvolve / upscale 2 times
                layer, weight = deconvolution(self.g_layers[-1], out_shape, i, name='g_h'+str(i))
                self.g_weights.append(weight)
                # batch normalization and activation
                self.g_layers.append(tf.nn.relu(self.g_bn_layers[i](layer, train)))

            # upscale
            layer, weight = deconvolution(self.g_layers[-1],
                                          [
                                              self.batch_size,
                                              (2**self.opt.num_layers)*(h-1)+1,
                                              (2**self.opt.num_layers)*(w-1)+1,
                                              self.g_filters[self.opt.num_layers - 1]
                                          ],
                                          self.opt.num_layers - 1,
                                          name='g_h'+str(self.opt.num_layers - 1))

            self.g_weights.append(weight)
            # activate without batch normalization
            self.g_layers.append(tf.nn.tanh(layer, name='output'))

            return self.g_layers[-1]

    def load(self, checkpoint_dir):

        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def hyper_param_summarization(self, h_dict):

        # define the list of important hyper_params
        for name, var in h_dict:
            self.hyper_params += str(name)+'='+str(var)

