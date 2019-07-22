import os
import tensorflow as tf
import numpy as np

from vgg19 import VGG19
from dataprocess import DataProcess


class Utils(object):
    def __init__(self,
                 args):
        self.args = args

    @staticmethod
    def read_image(image_path):
        image = tf.gfile.GFile(image_path, 'rb').read()
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.to_float(image)

        return image

    def get_feature_map(self):
        dataprocess = DataProcess(self.args)

        image = self.read_image(self.args.style_image_path)
        image = tf.image.resize_images(image, [self.args.image_size, self.args.image_size],
                                       tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.expand_dims(image, 0)
        image = dataprocess.mean_image_subtraction(image)

        vgg = VGG19(self.args)
        vgg_net = vgg.feed_forward(image, reuse=True)

        with tf.Session() as sess:
            sess.run((tf.global_variables_initializer()))
            vgg_net_ = sess.run(vgg_net)

        return vgg_net_

    @staticmethod
    def get_style_conv(vgg_net):
        return [vgg_net['conv1_1'], vgg_net['conv2_1'],
                vgg_net['conv3_1'], vgg_net['conv4_1'], vgg_net['conv5_1']]

    @staticmethod
    def get_noise(shape):
        return np.random.uniform(-1., 1., size=shape)

    @staticmethod
    def get_var_list(scope_list):
        var_list = []

        for scope in scope_list:
            var_list.extend(tf.trainable_variables(scope=scope))

        return var_list


