import os
import time
import numpy as np

import tensorflow as tf

from generator import Generators
from dataloder import Dataset
from vgg19 import VGG19
from loss import Loss
from dataprocess import DataProcess
from utils import Utils

class Solver(object):
    def __init__(self,
                 args):
        self.args = args

    def train(self):
        Vgg = VGG19(self.args)
        generators = Generators(self.args)
        dataprocess = DataProcess(self.args)
        utils = Utils(self.args)

        # define the placeholder for filename list of data
        data_list = tf.placeholder(dtype=tf.string, shape=[None])
        # define the placeholder for the input
        style_image = tf.placeholder(dtype=tf.float32, shape=[self.args.batch_size, self.args.image_size, self.args.image_size, 3])
        origin_image = tf.placeholder(dtype=tf.float32, shape=[self.args.batch_size, self.args.image_size, self.args.image_size, 3])

        # load the dataset
        dataset = Dataset(self.args)
        iterator = dataset.load_dataset(data_list)
        img1, img2 = iterator.get_next()

        processed_style_image = dataprocess.mean_image_subtraction(style_image)
        remove_style_image = generators.style_remover(processed_style_image, False)

        processed_remove_style_image = dataprocess.mean_image_subtraction(remove_style_image)
        processed_origin_image = dataprocess.mean_image_subtraction(origin_image)

        loss = Loss(self.args)
        # define the style transfer loss
        vgg_source = Vgg.feed_forward(processed_origin_image, reuse=False)
        vgg_generate = Vgg.feed_forward(processed_remove_style_image, reuse=True)
        loss_content = loss.content_loss(vgg_source['conv4_2'], vgg_generate['conv4_2'])

        sum_loss_content = tf.summary.scalar('loss_content', loss_content)

        sum_style_image = tf.summary.image('Source/style_image', style_image)
        sum_origin_image = tf.summary.image('Source/origin_image', origin_image)
        sum_generate_image = tf.summary.image('Generate/generate_image', remove_style_image)

        summary_loss = tf.summary.merge([sum_loss_content])
        summary_image = tf.summary.merge([sum_style_image, sum_origin_image, sum_generate_image])

        writer = tf.summary.FileWriter(self.args.log_path)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)

        var_trained = utils.get_var_list(['subsampled', 'upsampling', 'style_remover'])
        var_saved = tf.trainable_variables()
        saver = tf.train.Saver(var_list=var_saved,
                               max_to_keep=5,
                               write_version=tf.train.SaverDef.V1)

        train_op = optimizer.minimize(loss_content, var_list=var_trained)

        init_global = tf.global_variables_initializer()

        step = 0
        with tf.Session() as sess:
            sess.run(init_global)
            sess.run(iterator.initializer, feed_dict={data_list: dataset.tfrecord})

            while True:
                try:
                    img1_, img2_ = sess.run([img1, img2])

                    _ = sess.run(train_op, feed_dict={style_image: img1_, origin_image: img2_})

                    if step % self.args.save_summary == 0:
                        _summary, _loss = sess.run([summary_loss, loss_content],
                                                   feed_dict={style_image: img1_, origin_image: img2_})
                        writer.add_summary(_summary, step)
                        print('Step: ', step, 'content_loss: ', _loss)
                        writer.flush()

                    if step % self.args.save_image == 0:
                        _summary_image = sess.run(summary_image, feed_dict={style_image: img1_, origin_image: img2_})
                        writer.add_summary(_summary_image, step)
                        writer.flush()

                    if step % self.args.save_epoch == 0:
                        saver.save(sess,
                                   os.path.join(self.args.model_path, f'style-remover.ckpt-{step}'))
                except tf.errors.OutOfRangeError:
                    print("End of training")
                    saver.save(sess,
                               os.path.join(self.args.model_path, 'style-remover.ckpt-done'))
                    break
                else:
                    step += 1

    def test_dir(self):
        generators = Generators(self.args)
        dataprocess = DataProcess(self.args)
        utils = Utils(self.args)

        origin_images = os.listdir(self.args.test_dir)

        for name in origin_images:
            image = utils.read_image(os.path.join(self.args.test_dir, name))
            image = tf.expand_dims(image, 0)
            image = dataprocess.mean_image_subtraction(image)

            with tf.Session() as sess:
                image_ = sess.run(image)

            generated = generators.style_remover(image_, reuse=False)

            generated = tf.cast(generated, tf.uint8)
            generated = tf.squeeze(generated, [0])

            saver = tf.train.Saver(tf.global_variables(),
                                   write_version=tf.train.SaverDef.V1)
            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, self.args.model_file)

                generated_file = os.path.join(self.args.save_dir, 'remove-style') + name

                with open(generated_file, 'wb') as img:
                    start_time = time.time()
                    img.write(sess.run(tf.image.encode_jpeg(generated)))
                    end_time = time.time()
                    print('Generated_file elapsed time: %fs' % (end_time - start_time))

            tf.reset_default_graph()


