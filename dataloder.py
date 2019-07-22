import os
import sys
import tensorflow as tf


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def parse_function(example_proto):
    keys_to_features = {
        'image/style_encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/origin_encoded': tf.FixedLenFeature((), tf.string, default_value='')
    }
    parsed_example = tf.parse_single_example(example_proto, keys_to_features)

    style_image = tf.image.decode_jpeg(parsed_example['image/style_encoded'], channels=3)
    origin_image = tf.image.decode_jpeg(parsed_example['image/origin_encoded'], channels=3)

    style_image = tf.to_float(style_image)
    origin_image = tf.to_float(origin_image)

    return style_image, origin_image


class Dataset(object):
    def __init__(self,
                 args):
        self.args = args
        self.tfrecord = []
        self.style_image_list = os.listdir(self.args.transfer_image_path)

        for tfrecord in os.listdir(self.args.dataSet):
            self.tfrecord.append(os.path.join(self.args.dataSet, tfrecord))

    def _add_to_tfrecord(self, filename, tfrecord_writer):
        style_image = tf.gfile.GFile(os.path.join(self.args.transfer_image_path, filename), 'rb').read()
        origin_image = tf.gfile.GFile(os.path.join(self.args.origin_image_path, filename[5:]), 'rb').read()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/style_encoded': bytes_feature(style_image),
            'image/origin_encoded': bytes_feature(origin_image)
        }))

        tfrecord_writer.write(example.SerializeToString())

    def create_dataset(self):
        file_created = 0
        file_saved = 0

        while file_created < self.args.tfrecord_num:
            tf_filename = '%s/train_%03d.tfrecord' % (self.args.dataSet,
                                                      file_saved)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                file_created_per_record = 0
                while file_created < self.args.tfrecord_num and file_created_per_record < self.args.samples_per_file:
                    sys.stdout.write('\r>> Converting image %d/%d' % (file_created, self.args.tfrecord_num))
                    sys.stdout.flush()
                    filename = self.style_image_list[file_created]
                    self._add_to_tfrecord(filename, tfrecord_writer)
                    file_created += 1
                    file_created_per_record += 1
                file_saved += 1

        print('\nFinished converting to the tfrecord.')

    def load_dataset(self, data_list):
        dataset = tf.data.TFRecordDataset(data_list)
        new_dataset = dataset.map(parse_function)

        # shuffle_dataset = new_dataset.shuffle(buffer_size=10000)
        batch_dataset = new_dataset.batch(self.args.batch_size)
        epoch_dataset = batch_dataset.repeat(self.args.num_epochs)

        iterator = epoch_dataset.make_initializable_iterator()

        return iterator

    def test_dataset(self):
        data_list = tf.placeholder(tf.string, shape=[None])

        iterator = self.load_dataset(data_list)
        style_image, origin_image = iterator.get_next()

        count = 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={data_list: self.tfrecord})

            while True:
                try:
                    style_image_, origin_image_ = sess.run([style_image, origin_image])
                except tf.errors.OutOfRangeError:
                    print("End of dataSet")
                    break
                else:
                    print("No.%d" % count)
                    print("style image shape: %s | type: %s" % (style_image_.shape, style_image_.dtype))
                    print("origin image shape: %s | type: %s" % (origin_image_.shape, origin_image_.dtype))
                count += 1
