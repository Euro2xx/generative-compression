""" Modular components of computational graph
    JTan 2018
"""
import tensorflow as tf
from utils import Utils
import numpy as np


def get_weight(shape, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight"):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)

    # Equalized learning rate and custom learning rate multiplier
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape = shape, initializer = init) * runtime_coef


def conv2d_layer(x, filters, kernel_size, up = False, down = False, resample_kernel = None, gain = 1,
        use_wscale = True, lrmul = 1, weight_var = "weight"):

    assert not (up and down)
    assert kernel_size >= 1 and kernel_size % 2 == 1
    dim = filters
    w = get_weight([kernel_size, kernel_size, tf.get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale,
        lrmul = lrmul, weight_var = weight_var)

    x = tf.nn.conv2d(x, w, data_format = "NCHW", strides = [1, 1, 1, 1], padding = "SAME")
    return x

def dense_layer(x, dim, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight", name = None):
    if name is not None:
        weight_var = "{}_{}".format(weight_var, name)

    w = get_weight([get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale,
        lrmul = lrmul, weight_var = weight_var)

    return tf.matmul(x, w)

def get_shape(x):
    shape, dyn_shape = x.shape.as_list().copy(), tf.shape(x)
    for index, dim in enumerate(shape):
        if dim is None:
            shape[index] = dyn_shape[index]
    return shape

def mlp(x, resnet, layers_num, dim, act, lrmul, pooling = "mean", transformer = False, norm_type = None, **kwargs):
    shape = get_shape(x)

    for layer_idx in range(layers_num):
        with tf.variable_scope("Dense%d" % layer_idx):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
            x = norm(x, norm_type)

    x = tf.reshape(x, [-1] + shape[1:-1] + [dim])
    return x

class Network(object):

    @staticmethod
    def encoder(x, config, training, C, reuse=False, actv=tf.nn.relu, scope='image'):
        """
        Process image x ([512,1024]) into a feature map of size W/16 x H/16 x C
         + C:       Bottleneck depth, controls bpp
         + Output:  Projection onto C channels, C = {2,4,8,16}
        """
        init = tf.contrib.layers.xavier_initializer()
        print('<------------ Building global {} generator architecture ------------>'.format(scope))

        def conv_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=actv, init=init):
            bn_kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
            in_kwargs = {'center':True, 'scale': True}
            x = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
            # x = tf.layers.batch_normalization(x, **bn_kwargs)
            x = tf.contrib.layers.instance_norm(x, **in_kwargs)
            x = actv(x)
            return x

        with tf.variable_scope('encoder_{}'.format(scope), reuse=reuse):

            # Run convolutions
            f = [60, 120, 240, 480, 960]
            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            out = conv_block(x, filters=f[0], kernel_size=7, strides=1, padding='VALID', actv=actv)

            out = conv_block(out, filters=f[1], kernel_size=3, strides=2, actv=actv)
            print("1 layer", tf.shape(out))
            out = conv_block(out, filters=f[2], kernel_size=3, strides=2, actv=actv)
            print("2 layer", tf.shape(out))
            out = conv_block(out, filters=f[3], kernel_size=3, strides=2, actv=actv)
            print("3 layer", tf.shape(out))
            out = conv_block(out, filters=f[4], kernel_size=3, strides=2, actv=actv)
            print("4 layer", tf.shape(out))
            # Project channels onto space w/ dimension C
            # Feature maps have dimension W/16 x H/16 x C
            out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')

            feature_map = conv_block(out, filters=C, kernel_size=3, strides=1, padding='VALID', actv=actv)
            print("feature map", tf.shape(feature_map))
            return feature_map


    @staticmethod
    def quantizer(w, config, reuse=False, temperature=1, L=5, scope='image'):
        """
        Quantize feature map over L centers to obtain discrete $\hat{w}$
         + Centers: {-2,-1,0,1,2}
         + TODO:    Toggle learnable centers?
        """
        with tf.variable_scope('quantizer_{}'.format(scope, reuse=reuse)):

            centers = tf.cast(tf.range(-2,3), tf.float32)
            # Partition W into the Voronoi tesellation over the centers
            w_stack = tf.stack([w for _ in range(L)], axis=-1)
            w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)

            smx = tf.nn.softmax(-1.0/temperature * tf.abs(w_stack - centers), dim=-1)
            # Contract last dimension
            w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))

            # Treat quantization as differentiable for optimization
            w_bar = tf.round(tf.stop_gradient(w_hard - w_soft) + w_soft)

            return w_bar


    @staticmethod
    def decoder(w_bar, config, training, C, reuse=False, actv=tf.nn.relu, channel_upsample=960):
        """
        Attempt to reconstruct the image from the quantized representation w_bar.
        Generated image should be consistent with the true image distribution while
        recovering the specific encoded image
        + C:        Bottleneck depth, controls bpp - last dimension of encoder output
        + TODO:     Concatenate quantized w_bar with noise sampled from prior
        """
        init = tf.contrib.layers.xavier_initializer()

        def residual_block(x, n_filters, kernel_size=3, strides=1, actv=actv):
            init = tf.contrib.layers.xavier_initializer()
            # kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
            strides = [1,1]
            identity_map = x

            p = int((kernel_size-1)/2)
            res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
            res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                    activation=None, padding='VALID')
            res = actv(tf.contrib.layers.instance_norm(res))

            res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
            res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                    activation=None, padding='VALID')
            res = tf.contrib.layers.instance_norm(res)

            assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between input/output!'
            out = tf.add(res, identity_map)

            return out

        def upsample_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=actv, batch_norm=False):
            bn_kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
            in_kwargs = {'center':True, 'scale': True}
            x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
            if batch_norm is True:
                x = tf.layers.batch_normalization(x, **bn_kwargs)
            else:
                x = tf.contrib.layers.instance_norm(x, **in_kwargs)
            x = actv(x)

            return x

        # Project channel dimension of w_bar to higher dimension
        # W_pc = tf.get_variable('W_pc_{}'.format(C), shape=[C, channel_upsample], initializer=init)
        # upsampled = tf.einsum('ijkl,lm->ijkm', w_bar, W_pc)
        with tf.variable_scope('decoder', reuse=reuse):
            w_bar = tf.pad(w_bar, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            upsampled = Utils.conv_block(w_bar, filters=960, kernel_size=3, strides=1, padding='VALID', actv=actv)
            
            # Process upsampled feature map with residual blocks
            res = residual_block(upsampled, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)

            # Upsample to original dimensions - mirror decoder
            f = [480, 240, 120, 60]

            ups = upsample_block(res, f[0], 3, strides=[2,2], padding='same')
            ups = upsample_block(ups, f[1], 3, strides=[2,2], padding='same')
            ups = upsample_block(ups, f[2], 3, strides=[2,2], padding='same')
            ups = upsample_block(ups, f[3], 3, strides=[2,2], padding='same')
            
            ups = tf.pad(ups, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            ups = tf.layers.conv2d(ups, 3, kernel_size=7, strides=1, padding='VALID')

            out = tf.nn.tanh(ups)

            return out


    @staticmethod
    def discriminator(x, config, training, reuse=False, actv=tf.nn.leaky_relu, use_sigmoid=False, ksize=4):
        # x is either generator output G(z) or drawn from the real data distribution
        # Patch-GAN discriminator based on arXiv 1711.11585
        # bn_kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        in_kwargs = {'center':True, 'scale':True, 'activation_fn':actv}

        print('Shape of x:', x.get_shape().as_list())

        with tf.variable_scope('discriminator', reuse=reuse):
            c1 = tf.layers.conv2d(x, 64, kernel_size=ksize, strides=2, padding='same', activation=actv)
            c2 = tf.layers.conv2d(c1, 128, kernel_size=ksize, strides=2, padding='same')
            c2 = actv(tf.contrib.layers.instance_norm(c2, **in_kwargs))
            c3 = tf.layers.conv2d(c2, 256, kernel_size=ksize, strides=2, padding='same')
            c3 = actv(tf.contrib.layers.instance_norm(c3, **in_kwargs))
            c4 = tf.layers.conv2d(c3, 512, kernel_size=ksize, strides=2, padding='same')
            c4 = actv(tf.contrib.layers.instance_norm(c4, **in_kwargs))

            out = tf.layers.conv2d(c4, 1, kernel_size=ksize, strides=1, padding='same')

            if use_sigmoid is True:  # Otherwise use LS-GAN
                out = tf.nn.sigmoid(out)

        return out


    @staticmethod
    def multiscale_discriminator(x, config, training, actv=tf.nn.leaky_relu, use_sigmoid=False,
        ksize=4, mode='real', reuse=False):
        # x is either generator output G(z) or drawn from the real data distribution
        # Multiscale + Patch-GAN discriminator architecture based on arXiv 1711.11585
        print('<------------ Building multiscale discriminator architecture ------------>')

        if mode == 'real':
            print('Building discriminator D(x)')
        elif mode == 'reconstructed':
            print('Building discriminator D(G(z))')
        else:
            raise NotImplementedError('Invalid discriminator mode specified.')

        # Downsample input
        x2 = tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='same')
        x4 = tf.layers.average_pooling2d(x2, pool_size=3, strides=2, padding='same')

        print('Shape of x:', x.get_shape().as_list())
        print('Shape of x downsampled by factor 2:', x2.get_shape().as_list())
        print('Shape of x downsampled by factor 4:', x4.get_shape().as_list())

        def discriminator(x, scope, actv=actv, use_sigmoid=use_sigmoid, ksize=ksize, reuse=reuse):

            # Returns patch-GAN output + intermediate layers

            with tf.variable_scope('discriminator_{}'.format(scope), reuse=reuse):
                c1 = tf.layers.conv2d(x, 64, kernel_size=ksize, strides=2, padding='same', activation=actv)
                c2 = Utils.conv_block(c1, filters=128, kernel_size=ksize, strides=2, padding='same', actv=actv)
                c3 = Utils.conv_block(c2, filters=256, kernel_size=ksize, strides=2, padding='same', actv=actv)
                c4 = Utils.conv_block(c3, filters=512, kernel_size=ksize, strides=2, padding='same', actv=actv)
                out = tf.layers.conv2d(c4, 1, kernel_size=ksize, strides=1, padding='same')

                if use_sigmoid is True:  # Otherwise use LS-GAN
                    out = tf.nn.sigmoid(out)

            return out, c1, c2, c3, c4

        with tf.variable_scope('discriminator', reuse=reuse):
            disc, *Dk = discriminator(x, 'original')
            disc_downsampled_2, *Dk_2 = discriminator(x2, 'downsampled_2')
            disc_downsampled_4, *Dk_4 = discriminator(x4, 'downsampled_4')

        return disc, disc_downsampled_2, disc_downsampled_4, Dk, Dk_2, Dk_4

    @staticmethod
    def dcgan_generator(z, config, training, C, reuse=False, actv=tf.nn.relu, kernel_size=5, upsample_dim=256):
        """
        Upsample noise to concatenate with quantized representation w_bar.
        + z:    Drawn from latent distribution - [batch_size, noise_dim]
        + C:    Bottleneck depth, controls bpp - last dimension of encoder output
        """
        init =  tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        with tf.variable_scope('noise_generator', reuse=reuse):

            # [batch_size, 4, 8, dim]
            with tf.variable_scope('fc1', reuse=reuse):
                h2 = tf.layers.dense(z, units=4 * 8 * upsample_dim, activation=actv, kernel_initializer=init)  # cifar-10
                h2 = tf.layers.batch_normalization(h2, **kwargs)
                h2 = tf.reshape(h2, shape=[-1, 4, 8, upsample_dim])

            # [batch_size, 8, 16, dim/2]
            with tf.variable_scope('upsample1', reuse=reuse):
                up1 = tf.layers.conv2d_transpose(h2, upsample_dim//2, kernel_size=kernel_size, strides=2, padding='same', activation=actv)
                up1 = tf.layers.batch_normalization(up1, **kwargs)

            # [batch_size, 16, 32, dim/4]
            with tf.variable_scope('upsample2', reuse=reuse):
                up2 = tf.layers.conv2d_transpose(up1, upsample_dim//4, kernel_size=kernel_size, strides=2, padding='same', activation=actv)
                up2 = tf.layers.batch_normalization(up2, **kwargs)

            # [batch_size, 32, 64, dim/8]
            with tf.variable_scope('upsample3', reuse=reuse):
                up3 = tf.layers.conv2d_transpose(up2, upsample_dim//8, kernel_size=kernel_size, strides=2, padding='same', activation=actv)  # cifar-10
                up3 = tf.layers.batch_normalization(up3, **kwargs)

            with tf.variable_scope('conv_out', reuse=reuse):
                out = tf.pad(up3, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
                out = tf.layers.conv2d(out, C, kernel_size=7, strides=1, padding='VALID')

        return out

    @staticmethod
    def dcgan_discriminator(x, config, training, reuse=False, actv=tf.nn.relu):
        # x is either generator output G(z) or drawn from the real data distribution
        init =  tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        print('Shape of x:', x.get_shape().as_list())
        x = tf.reshape(x, shape=[-1, 32, 32, 3])
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1', reuse=reuse):
                c1 = tf.layers.conv2d(x, 64, kernel_size=5, strides=2, padding='same', activation=actv)
                c1 = tf.layers.batch_normalization(c1, **kwargs)

            with tf.variable_scope('conv2', reuse=reuse):
                c2 = tf.layers.conv2d(c1, 128, kernel_size=5, strides=2, padding='same', activation=actv)
                c2 = tf.layers.batch_normalization(c2, **kwargs)

            with tf.variable_scope('fc1', reuse=reuse):
                fc1 = tf.contrib.layers.flatten(c2)
                # fc1 = tf.reshape(c2, shape=[-1, 8 * 8 * 128])
                fc1 = tf.layers.dense(fc1, units=1024, activation=actv, kernel_initializer=init)
                fc1 = tf.layers.batch_normalization(fc1, **kwargs)

            with tf.variable_scope('out', reuse=reuse):
                out = tf.layers.dense(fc1, units=2, activation=None, kernel_initializer=init)

        return out

