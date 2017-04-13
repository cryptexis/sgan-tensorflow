import os
import time
import numpy as np
from utils.data_util import get_data, save_tensor
from model.ops import *


class SpatialGan(object):

    real_images = None
    random_noise = None
    E_net = None
    E_loss = None

    E_vars = None
    G_vars = None
    # manifold
    GM_net = None
    GM_loss = None
    DM_net_real = None
    DM_logits_real = None
    DM_loss = None
    DM_loss_real = None
    DM_loss_fake = None
    DM_net_fake = None
    DM_logits_fake = None
    DM_vars = None
    # diffusion
    GD_net = None
    GD_loss = None
    DD_net_real = None
    DD_logits_real = None
    DD_loss = None
    DD_loss_real = None
    DD_loss_fake = None
    DD_net_fake = None
    DD_logits_fake = None
    DD_vars = None

    lmr_sum = None
    lmf_sum = None
    lmg_sum = None
    lmt_sum = None
    img_sum_m = None

    ldr_sum = None
    ldf_sum = None
    ldg_sum = None
    ldt_sum = None
    img_sum_d = None

    le_sum = None

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
        self.d1_layers = []
        self.d1_filters = [2**(n+6) for n in range(self.opt.num_layers-1)]+[1]
        self.d1_weights = []

        self.d2_layers = []
        self.d2_filters = [2**(n+6) for n in range(self.opt.num_layers-1)]+[1]
        self.d2_weights = []
        # encoder
        self.e_layers = []
        self.e_filters = [2**(n+6) for n in range(self.opt.num_layers-1)]+[self.opt.z_dim]
        self.e_weights = []

        # same for the G_network
        self.g_layers = []
        self.g_weights = []
        self.g_filters = [3] + [2 ** (n + 6) for n in range(self.opt.num_layers - 1)]
        self.g_filters = self.g_filters[::-1]

        self.d1_bn_layers = []
        self.d2_bn_layers = []
        self.e_bn_layers = []
        self.g_bn_layers = []

        # batch normalization layers
        for i in range(0, self.opt.num_layers - 2):
            self.d1_bn_layers.append(BatchNorm(name='d1_bn'+str(i+1)))
        for i in range(0, self.opt.num_layers - 2):
            self.d2_bn_layers.append(BatchNorm(name='d2_bn'+str(i+1)))

        for i in range(0, self.opt.num_layers - 2):
            self.e_bn_layers.append(BatchNorm(name='e_bn'+str(i+1)))

        for i in range(0, self.opt.num_layers - 1):
            self.g_bn_layers.append(BatchNorm(name='g_bn'+str(i+1)))

    def build_model(self):

        self.real_images = tf.placeholder(tf.float32,
                                          [self.batch_size] + self.image_shape,
                                          name='real_images')

        self.random_noise = tf.placeholder(tf.float32,
                                           [self.batch_size] + self.z_dim,
                                           name='z')
        if self.opt.is_train:
            # MANIFOLD
            # forward pass through generator
            # passing real data through encoder
            self.E_net = self.encoder(self.real_images)  # 9x9x100
            self.GM_net = self.generator(self.E_net)
            # forward pass through manifold discriminator / real images
            self.DM_net_real, self.DM_logits_real = self.discriminator_manifold(self.real_images)
            # forward pass through discriminator generated data on manifold
            self.DM_net_fake, self.DM_logits_fake = self.discriminator_manifold(self.GM_net, reuse=True)

            # DIFFUSION
            self.GD_net = self.generator(self.random_noise, reuse=True)
            self.DD_net_real, self.DD_logits_real = self.discriminator_diffusion(self.GM_net)
            self.DD_net_fake, self.DD_logits_fake = self.discriminator_diffusion(self.GD_net, reuse=True)
        else:
            self.GD_net = self.generator(self.random_noise)

        if self.opt.is_train:
            # MANIFOLD STEP LOSS
            # DISCRIMINATOR
            self.DM_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DM_logits_real,
                                                        labels=tf.ones_like(self.DM_logits_real)))
            self.lmr_sum = tf.summary.scalar("DM_loss_real", self.DM_loss_real)

            self.DM_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DM_logits_fake,
                                                        labels=tf.zeros_like(self.DM_logits_fake)))

            self.lmf_sum = tf.summary.scalar("DM_loss_fake", self.DM_loss_fake)
            # ENCODER
            self.E_loss = tf.reduce_mean(tf.square(self.GM_net - self.real_images))
            # GENERATOR
            self.GM_loss = 0.005*tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DM_logits_fake,
                                                            labels=tf.ones_like(self.DM_logits_fake))) + self.E_loss

            self.lmg_sum = tf.summary.scalar("GM_loss", self.GM_loss)
            self.img_sum_m = tf.summary.image("output_manifold", self.GM_net, 3)

            # Discriminative loss + L2 decay on weights
            self.DM_loss = self.DM_loss_real + self.DM_loss_fake
            self.lmt_sum = tf.summary.scalar("DM_loss", self.DM_loss)

            # DIFFUSION STEP LOSS
            # DISCRIMINATOR
            self.DD_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DD_logits_real,
                                                        labels=tf.ones_like(self.DD_logits_real)))
            self.ldr_sum = tf.summary.scalar("DD_loss_real", self.DD_loss_real)

            self.DD_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DD_logits_fake,
                                                        labels=tf.zeros_like(self.DD_logits_fake)))
            self.ldf_sum = tf.summary.scalar("DD_loss_fake", self.DD_loss_fake)

            self.GD_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DD_logits_fake,
                                                        labels=tf.ones_like(self.DD_logits_fake)))

            self.ldg_sum = tf.summary.scalar("GD_loss", self.GD_loss)
            self.img_sum_d = tf.summary.image("output_diffusion", self.GD_net, 3)

            # Discriminative loss + L2 decay on weights
            self.DD_loss = self.DD_loss_real + self.DD_loss_fake
            self.ldt_sum = tf.summary.scalar("DD_loss", self.DD_loss)

            self.le_sum = tf.summary.scalar("E_loss", self.E_loss)

            t_vars = tf.trainable_variables()

            self.DM_vars = [var for var in t_vars if 'd1_' in var.name]
            self.DD_vars = [var for var in t_vars if 'd2_' in var.name]
            self.E_vars = [var for var in t_vars if 'e_' in var.name]
            self.G_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):

        # define optimizers for all networks
        dm_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1=self.opt.beta1) \
            .minimize(self.DM_loss, var_list=self.DM_vars)

        gm_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1=self.opt.beta1) \
            .minimize(self.GM_loss, var_list=self.G_vars)

        dd_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1=self.opt.beta1) \
            .minimize(self.DD_loss, var_list=self.DD_vars)

        gd_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1=self.opt.beta1) \
            .minimize(self.GD_loss, var_list=self.G_vars)

        e_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1=self.opt.beta1) \
            .minimize(self.E_loss, var_list=self.E_vars)

        tf.global_variables_initializer().run()

        gm_sum = tf.summary.merge([self.lmg_sum, self.img_sum_m])
        gd_sum = tf.summary.merge([self.ldg_sum, self.img_sum_d])

        dm_sum = tf.summary.merge([self.lmr_sum, self.lmf_sum, self.lmt_sum])
        dd_sum = tf.summary.merge([self.ldr_sum, self.ldf_sum, self.ldt_sum])
        e_sum = tf.summary.merge([self.le_sum])

        sample_image = np.random.uniform(-1, 1, [self.opt.batch_size]+self.z_dim).astype(np.float32)
        writer = tf.summary.FileWriter(self.opt.checkpoint_dir, self.sess.graph)

        start_time = time.time()
        epoch = 0

        while epoch < self.opt.max_iterations:

            batch_images = get_data(self.opt.dataset, self.image_size, self.batch_size)
            batch_z = np.random.uniform(-1, 1, [self.opt.batch_size]+self.z_dim).astype(np.float32)
            if epoch % 3 == 0:
                print "Manifold Step: "
                # Update D network
                _, summary_str_dm = self.sess.run([dm_optim, dm_sum],
                                                  feed_dict={
                                                       self.real_images: batch_images,
                                                   })
                # Update G network
                _, summary_str_gm = self.sess.run([gm_optim, gm_sum],
                                                  feed_dict={
                                                       self.real_images: batch_images,
                                                       self.random_noise: batch_z
                                               })

                writer.add_summary(summary_str_dm, epoch)
                writer.add_summary(summary_str_gm, epoch)
            elif epoch % 3 == 1:
                print "Diffusion Step:"
                # Update D network
                _, summary_str_dd = self.sess.run([dd_optim, dd_sum],
                                                  feed_dict={
                                                       self.real_images: batch_images,
                                                       self.random_noise: batch_z
                                                   })

                # Update G network
                _, summary_str_gd = self.sess.run([gd_optim, gd_sum],
                                                  feed_dict={
                                                       self.random_noise: batch_z
                                                   })
                writer.add_summary(summary_str_dd, epoch)
                writer.add_summary(summary_str_gd, epoch)
            else:

                print "Encoder Step: "
                # Update D network
                _, summary_str_e = self.sess.run([e_optim, e_sum],
                                                 feed_dict={
                                                        self.real_images: batch_images,
                                                    })
                writer.add_summary(summary_str_e, epoch)

            if epoch % self.opt.checkpoint_interval == 0:

                img_out = self.GD_net.eval({self.random_noise: sample_image})
                save_tensor(img_out[0], os.path.join(self.opt.sample_dir, str(epoch)+'.jpg'))

            if epoch % 50 == 0:
                self.saver.save(self.sess, os.path.join(self.opt.checkpoint_dir, 'fck.ckpt'), global_step=epoch)

            time_delta = time.time() - start_time
            print("Epoch: [%2d] time: %4.4f" % (epoch, time_delta))

            epoch += 1

    def encoder(self, image, reuse=False):

        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()
            # down sample the first layer
            conv, weight = convolution(image, self.e_filters[0], name='e_h0_conv')

            if not reuse:
                self.e_weights.append(weight)
            self.e_layers.append(lrelu(conv))

            for i in range(0, self.opt.num_layers - 2):
                conv, weight = convolution(self.e_layers[-1],
                                           self.e_filters[i+1],
                                           name='e_h'+str(i+1)+'_conv')
                if not reuse:
                    self.e_weights.append(weight)
                self.e_layers.append(
                    lrelu(
                        self.e_bn_layers[i](conv)
                    )
                )
            # last layer
            layer, weight = convolution(self.e_layers[-1], self.e_filters[-1], name='e_h4_conv')
            if not reuse:
                self.e_weights.append(weight)
            self.e_layers.append(tf.nn.tanh(layer, name='output'))

        return self.e_layers[-1]

    def discriminator_manifold(self, image, reuse=False):
        """
        In discriminator network batch normalization
        used on all layers except input and output
        :param image:
        :param reuse:
        :return:
        """
        with tf.variable_scope("discriminator_manifold") as scope:
            if reuse:
                scope.reuse_variables()
            # down sample the first layer
            conv, weight = convolution(image, self.d1_filters[0], name='d1_h0_conv')

            if not reuse:
                self.d1_weights.append(weight)
            self.d1_layers.append(lrelu(conv))

            for i in range(0, self.opt.num_layers - 2):
                conv, weight = convolution(self.d1_layers[-1],
                                           self.d1_filters[i+1],
                                           name='d1_h'+str(i+1)+'_conv')
                if not reuse:
                    self.d1_weights.append(weight)
                self.d1_layers.append(
                    lrelu(
                        self.d1_bn_layers[i](conv)
                    )
                )
            # last layer
            logit, weight = convolution(self.d1_layers[-1], self.d1_filters[-1], name='d1_h4_conv')
            if not reuse:
                self.d1_weights.append(weight)
            self.d1_layers.append(tf.nn.sigmoid(logit))

        return self.d1_layers[-1], logit

    def discriminator_diffusion(self, image, reuse=False):
        """
        In discriminator network batch normalization
        used on all layers except input and output
        :param image:
        :param reuse:
        :return:
        """
        with tf.variable_scope("discriminator_diffusion") as scope:
            if reuse:
                scope.reuse_variables()
            # down sample the first layer
            conv, weight = convolution(image, self.d2_filters[0], name='d2_h0_conv')

            if not reuse:
                self.d2_weights.append(weight)
            self.d2_layers.append(lrelu(conv))

            for i in range(0, self.opt.num_layers - 2):
                conv, weight = convolution(self.d2_layers[-1],
                                           self.d2_filters[i+1],
                                           name='d2_h'+str(i+1)+'_conv')
                if not reuse:
                    self.d2_weights.append(weight)
                self.d2_layers.append(
                    lrelu(
                        self.d2_bn_layers[i](conv)
                    )
                )
            # last layer
            logit, weight = convolution(self.d2_layers[-1], self.d2_filters[-1], name='d2_h4_conv')
            if not reuse:
                self.d2_weights.append(weight)
            self.d2_layers.append(tf.nn.sigmoid(logit))

        return self.d2_layers[-1], logit

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

                if not reuse:
                    self.g_weights.append(weight)
                # batch normalization and activation
                self.g_layers.append(lrelu(self.g_bn_layers[i](layer, train)))

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
            if not reuse:
                self.g_weights.append(weight)
            # activate without batch normalization
            self.g_layers.append(tf.nn.tanh(layer, name='output'))

            return self.g_layers[-1]

    def load(self, checkpoint_dir):
        import re
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
