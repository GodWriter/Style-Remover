import tensorflow as tf


class Layers(object):
    @staticmethod
    def conv2d(inputs, in_channels, out_channels, kernel, strides, padding, name):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(scope.name + '_w',
                                      [kernel, kernel, in_channels, out_channels],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable(scope.name + '_b',
                                     [out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            conv = tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding=padding)
            conv = tf.nn.bias_add(conv, biases)

            return conv

    @staticmethod
    def resize_conv2d(x, in_channels, out_channels, kernel, strides, padding, name):
        # get the origin shape
        origin_shape = x.get_shape().as_list()
        height, width = origin_shape[1], origin_shape[2]

        # compute the output shape
        new_height, new_width = height*strides*strides, width*strides*strides

        # Get the temporal image
        x_resized = tf.image.resize_images(x, [new_height, new_width],
                                           tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return Layers.conv2d(x_resized, in_channels, out_channels, kernel, strides, padding, name)

    @staticmethod
    def linear(x, output_size, name):
        input_size = x.get_shape().as_list()[1]

        with tf.variable_scope(name) as scope:
            matrix = tf.get_variable(scope.name + '_w',
                                     [input_size, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.2))
            biases = tf.get_variable(scope.name + '_b',
                                     [output_size],
                                     initializer=tf.constant_initializer(0.1))

            linear = tf.matmul(x, matrix) + biases

            return linear

    @staticmethod
    def residual(x, filters, kernel, strides, padding, name):
        with tf.variable_scope(name):
            conv1 = Layers.conv2d(x, filters, filters, kernel, strides, padding, 'conv1')
            relu_ = Layers.relu(conv1)
            conv2 = Layers.conv2d(relu_, filters, filters, kernel, strides, padding, 'conv2')

            residual = x + conv2

        return residual

    @staticmethod
    def instance_norm(x):
        epsilon = 1e-9
        # get each channel's mean and var
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

        # there should be matrix compute with  matrix
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    @staticmethod
    def relu(x):
        relu = tf.nn.relu(x)
        # convert nan to zero (nan != nan)
        nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))

        return nan_to_zero

    @staticmethod
    def leaky_relu(x):
        return tf.nn.leaky_relu(x)

    @staticmethod
    def pad(image):
        return tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    @staticmethod
    def remove_pad(y):
        shape = y.get_shape().as_list()
        height, width = shape[1], shape[2]

        y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

        return y

    @staticmethod
    def conv2d_spectral_norm(inputs, in_channels, out_channels, kernel, strides, padding, name, spectral_norm=True, update_collection=None):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(scope.name + '_w',
                                      [kernel, kernel, in_channels, out_channels],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable(scope.name + '_b',
                                     [out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            if spectral_norm:
                weights = weights_spectral_norm(weights, update_collection=update_collection)

            conv = tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding=padding)
            conv = tf.nn.bias_add(conv, biases)

            return conv

    @staticmethod
    def linear_spectral_norm(x, output_size, name, spectral_norm=True, update_collection=None):
        input_size = x.get_shape().as_list()[1]

        with tf.variable_scope(name) as scope:
            matrix = tf.get_variable(scope.name + '_w',
                                     [input_size, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.2))
            biases = tf.get_variable(scope.name + '_b',
                                     [output_size],
                                     initializer=tf.constant_initializer(0.1))

            if spectral_norm:
                matrix = weights_spectral_norm(matrix, update_collection=update_collection)

            linear = tf.matmul(x, matrix) + biases

            return linear

    @staticmethod
    def apply_noise(x, name='_w', noise_var=None, randomize_noise=True):
        with tf.variable_scope('Noise') as scope:
            if randomize_noise:
                noise = tf.random_normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1], dtype=x.dtype)
            else:
                noise = tf.cast(noise_var, x.dtype)
            weight = tf.get_variable(scope.name + name,
                                     [x.shape[3].value],
                                     dtype=x.dtype,
                                     initializer=tf.initializers.ones())
            return x + noise * tf.reshape(weight, [1, 1, 1, -1])


# spectral_norm
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm


def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            # 首先，u初始化为随机的噪音向量
            # u第一次是以None的形式传入，但是之后若是reuse后，u就共享了，不管是否为none，都是上一次的值
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                                trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1

        u_hat, v_hat, _ = power_iteration(u, iteration)

        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))

        w_mat = w_mat / sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not (update_collection == 'NO_OPS'):
                tf.add_to_collection(update_collection, u.assign(u_hat))

            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm

