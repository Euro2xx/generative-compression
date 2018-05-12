#!/usr/bin/python3
    
import tensorflow as tf
import numpy as np
import glob, time, os

from network import Network
from data import Data
from config import directories
from utils import Utils

class Model():
    def __init__(self, config, paths, name='gan_compression', evaluate=False):
        # Build the computational graph
        
        print('Building computational graph ...')
        self.G_global_step = tf.Variable(0, trainable=False)
        self.D_global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)

        # >>> Data handling
        self.path_placeholder = tf.placeholder(paths.dtype, paths.shape)
        self.test_path_placeholder = tf.placeholder(paths.dtype)

        train_dataset = Data.load_dataset(self.path_placeholder,
                                          config.batch_size,
                                          augment=False,
                                          multiscale=config.multiscale)
        test_dataset = Data.load_dataset(self.test_path_placeholder,
                                         config.batch_size,
                                         augment=False,
                                         multiscale=config.multiscale,
                                         test=True)

        self.iterator = tf.data.Iterator.from_string_handle(self.handle,
                                                                    train_dataset.output_types,
                                                                    train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()

        if config.multiscale:
            self.example, self.example_downscaled2, self.example_downscaled4 = self.iterator.get_next()
        else:
            self.example = self.iterator.get_next()            

        # noise_prior = tf.contrib.distributions.Uniform(-1., 1.)
        # self.noise_sample = noise_prior.sample([tf.shape(self.example)[0], config.noise_dim])

        # Global generator: Encode -> quantize -> reconstruct
        # =======================================================================================================>>>
        with tf.variable_scope('generator'):
            self.feature_map = Network.encoder(self.example, config, self.training_phase, config.channel_bottleneck)
            self.w_hat = Network.quantizer(self.feature_map, config)

            if config.sample_noise is True:
                print('Sampling noise...')
                noise_prior = tf.contrib.distributions.MultivariateNormalDiag(loc=tf.zeros([config.noise_dim]), scale_diag=tf.ones([config.noise_dim]))
                v = noise_prior.sample(tf.shape(self.example)[0])
                Gv = Network.dcgan_generator(v, config, self.training_phase, C=config.channel_bottleneck, upsample_dim=config.upsample_dim)
                self.z = tf.concat([self.w_hat, Gv], axis=-1)
            else:
                self.z = self.w_hat

            self.reconstruction = Network.decoder(self.z, config, self.training_phase, C=config.channel_bottleneck)
            # self.reconstruction = Network.decoder(self.feature_map, config, self.training_phase, config.channel_bottleneck)

        print('Real image shape:', self.example.get_shape().as_list())
        print('Reconstruction shape:', self.reconstruction.get_shape().as_list())

        # Pass generated, real images to discriminator
        # =======================================================================================================>>>
        if config.multiscale:
            D_x, D_x2, D_x4 = Network.multiscale_discriminator(self.example, self.example_downscaled2, self.example_downscaled4,
                self.reconstruction, config, self.training_phase, use_sigmoid=config.use_vanilla_GAN, mode='real')
            D_Gz, D_Gz2, D_Gz4 = Network.multiscale_discriminator(self.example, self.example_downscaled2, self.example_downscaled4,
                self.reconstruction, config, self.training_phase, use_sigmoid=config.use_vanilla_GAN, mode='reconstructed', reuse=True)
        else:
            D_x = Network.discriminator(self.example, config, self.training_phase, use_sigmoid=config.use_vanilla_GAN)
            D_Gz = Network.discriminator(self.reconstruction, config, self.training_phase, use_sigmoid=config.use_vanilla_GAN, reuse=True)
         
        # Loss terms 
        # =======================================================================================================>>>
        if config.use_vanilla_GAN is True:
            # Minimize JS divergence
            D_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_x,
                labels=tf.ones_like(D_x)))
            D_loss_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_Gz,
                labels=tf.zeros_like(D_Gz)))
            self.D_loss = D_loss_real + D_loss_gen
            # G_loss = max log D(G(z))
            self.G_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_Gz,
                labels=tf.ones_like(D_Gz)))
        else:
            # Minimize $\chi^2$ divergence
            self.D_loss = tf.reduce_mean(tf.square(D_x - 1.) + tf.square(D_Gz))
            self.G_loss = tf.reduce_mean(tf.square(D_Gz - 1.))

            if config.multiscale:
                self.D_loss += tf.reduce_mean(tf.square(D_x2 - 1.) + tf.square(D_x4 - 1.))
                self.D_loss += tf.reduce_mean(tf.square(D_Gz2) + tf.square(D_Gz4))

        self.G_loss += config.lambda_X * tf.losses.mean_squared_error(self.example, self.reconstruction)
        
        # Optimization
        # =======================================================================================================>>>
        G_opt = tf.train.AdamOptimizer(learning_rate=config.G_learning_rate, beta1=0.5)
        D_opt = tf.train.AdamOptimizer(learning_rate=config.D_learning_rate, beta1=0.5)

        theta_G = Utils.scope_variables('generator')
        theta_D = Utils.scope_variables('discriminator')
        print('Generator parameters:', theta_G)
        print('Discriminator parameters:', theta_D)
        G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        # Execute the update_ops before performing the train_step
        with tf.control_dependencies(G_update_ops):
            self.G_opt_op = G_opt.minimize(self.G_loss, name='G_opt', global_step=self.G_global_step, var_list=theta_G)
        with tf.control_dependencies(D_update_ops):
            self.D_opt_op = D_opt.minimize(self.D_loss, name='D_opt', global_step=self.D_global_step, var_list=theta_D)

        G_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.G_global_step)
        G_maintain_averages_op = G_ema.apply(theta_G)
        D_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.D_global_step)
        D_maintain_averages_op = D_ema.apply(theta_D)

        with tf.control_dependencies(G_update_ops+[self.G_opt_op]):
            self.G_train_op = tf.group(G_maintain_averages_op)
        with tf.control_dependencies(D_update_ops+[self.D_opt_op]):
            self.D_train_op = tf.group(D_maintain_averages_op)

        # >>> Monitoring
        # tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('generator_loss', self.G_loss)
        tf.summary.scalar('discriminator_loss', self.D_loss)
        tf.summary.scalar('G_global_step', self.G_global_step)
        tf.summary.scalar('D_global_step', self.D_global_step)
        tf.summary.image('real_images', self.example, max_outputs=1)
        tf.summary.image('compressed_images', self.reconstruction, max_outputs=1)
        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_train_{}'.format(name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_test_{}'.format(name, time.strftime('%d-%m_%I:%M'))))