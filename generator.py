import tensorflow as tf

from layers import Layers


class Generators(object):
    def __init__(self,
                 args):
        self.args = args

        # used for the information increment network
        self.INFO_WEIGHT = 10

    @staticmethod
    def subsampled(inputs, reuse=False):
        # Less border effect
        inputs = Layers.pad(inputs)

        with tf.variable_scope('subsampled', reuse=reuse):
            conv1 = Layers.conv2d(inputs, 3, 32, 9, 1, 'SAME', 'conv1')
            norm1 = Layers.instance_norm(conv1)
            relu1 = Layers.relu(norm1)

            conv2 = Layers.conv2d(relu1, 32, 64, 3, 2, 'SAME', 'conv2')
            norm2 = Layers.instance_norm(conv2)
            relu2 = Layers.relu(norm2)

            conv3 = Layers.conv2d(relu2, 64, 128, 3, 2, 'SAME', 'conv3')
            norm3 = Layers.instance_norm(conv3)
            relu3 = Layers.relu(norm3)

        return relu3

    @ staticmethod
    def residual_net(inputs, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            res1 = Layers.residual(inputs, 128, 3, 1, 'SAME', 'res1')
            res2 = Layers.residual(res1, 128, 3, 1, 'SAME', 'res2')
            res3 = Layers.residual(res2, 128, 3, 1, 'SAME', 'res3')
            res4 = Layers.residual(res3, 128, 3, 1, 'SAME', 'res4')
            res5 = Layers.residual(res4, 128, 3, 1, 'SAME', 'res5')
            # res6 = Layers.residual(res5, 128, 3, 1, 'SAME', 'res6')

        return res5

    @staticmethod
    def upsampling(inputs, reuse=False):
        with tf.variable_scope('upsampling', reuse=reuse):
            deconv1 = Layers.resize_conv2d(inputs, 128, 64, 3, 2, 'SAME', 'deconv1')
            denorm1 = Layers.instance_norm(deconv1)
            derelu1 = Layers.relu(denorm1)

            deconv2 = Layers.resize_conv2d(derelu1, 64, 32, 3, 2, 'SAME', 'deconv2')
            denorm2 = Layers.instance_norm(deconv2)
            derelu2 = Layers.relu(denorm2)

            deconv3 = Layers.resize_conv2d(derelu2, 32, 3, 9, 1, 'SAME', 'deconv3')
            denorm3 = Layers.instance_norm(deconv3)
            detanh3 = tf.nn.tanh(denorm3)

            y = (detanh3 + 1) * 127.5

            # Remove the border effect
            y = Layers.remove_pad(y)

        return y

    @staticmethod
    def style_remover(inputs, reuse=False):
        outputs = Generators.subsampled(inputs, False)
        outputs = Generators.residual_net(outputs, 'style_remover', False)
        outputs = Generators.upsampling(outputs, False)

        return outputs
