import argparse
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Thanks to
# https://towardsdatascience.com/how-to-build-efficient-audio-data-pipelines-with-tensorflow-2-0-b3133474c3c1
# for making this easy to do

_BASE_DIR = os.path.dirname('/home/car-sable/LaneDetection/')

_DEFAULT_META_CSV = os.path.join(_BASE_DIR, 'train_all.csv')
_DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, 'tf_records/')

_DEFAULT_TEST_SIZE = 0.1
_DEFAULT_VAL_SIZE = 0.1

# For a 4gb dataset, this makes it so that the avg shard is about 100 mb
_DEFAULT_NUM_SHARDS_TRAIN = 32
_DEFAULT_NUM_SHARDS_TEST = 4
_DEFAULT_NUM_SHARDS_VAL = 4

# _DEFAULT_NUM_SHARDS_TRAIN = 8
# _DEFAULT_NUM_SHARDS_TEST = 1
# _DEFAULT_NUM_SHARDS_VAL = 1



_SEED = 67083

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsConverter:
    """Convert images to TFRecords."""
    def __init__(self, meta, output_dir, n_shards_train, n_shards_test,
                 n_shards_val, test_size, val_size):
        self.output_dir = output_dir
        self.n_shards_train = n_shards_train
        self.n_shards_test = n_shards_test
        self.n_shards_val = n_shards_val

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        df = pd.read_csv(meta)

        # Shuffle data by "sampling" the entire data-frame
        self.df = df.sample(frac=1, random_state=_SEED)

        n_samples = len(df)
        # test_size and val_size is between 0 and 1
        # n_test is an integer
        self.n_test = math.ceil(n_samples * test_size)
        self.n_val = math.ceil(n_samples * val_size)
        self.n_train = n_samples - self.n_test - self.n_val

    def _get_shard_path(self, split, shard_id, shard_size):
        # the :03d is for zero padding
        return os.path.join(self.output_dir, f'{split}-{shard_id:03d}-{shard_size}.tfrecord')

    def _write_tfrecord_file(self, shard_path, indices):
        """Write TFRecord file."""

        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for index in indices:
                gt_file_path = self.df.Ground_Truth.iloc[index]
                binary_file_path = self.df.binary_image.iloc[index]
                seg_file_path = self.df.seg_label.iloc[index]

                # Example is a flexible message type that contains key-value
                # pairs, where each key maps to a Feature message. Here, each
                # Example contains two features: A FloatList for the decoded
                # audio data and an Int64List containing the corresponding
                # label's index.
                gt_image = Image.open(gt_file_path, "r")
                binary_image = Image.open(binary_file_path,"r")
                seg_file_image = Image.open(seg_file_path, "r")
#assumes the input and label images are the same size, if want different sizes should get all of them and store seperately
                gt_im_h = gt_image.height
                gt_im_w = gt_image.width
                gt_image_raw = gt_image.tobytes()
                binary_image_raw = binary_image.tobytes()
                seg_file_raw = seg_file_image.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'Height': _int64_feature(gt_im_h),
                    'Width': _int64_feature(gt_im_w),
                    'Ground_Truth': _bytes_feature(gt_image_raw),
                    'binary_image': _bytes_feature(binary_image_raw),
                    'seg_image': _bytes_feature(seg_file_raw)
                    }
                    ))

                out.write(example.SerializeToString())

    def convert(self):
        """Convert to TFRecords.
        Partition data into training, testing and validation sets. Then,
        divide each data set into the specified number of TFRecords shards.
        """
        splits = ('train', 'test', 'validate')
        split_sizes = (self.n_train, self.n_test, self.n_val)
        split_n_shards = (self.n_shards_train, self.n_shards_test,
                          self.n_shards_val)

        offset = 0
        for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
            print('Converting {} set into TFRecord shards...'.format(split))
            shard_size = math.ceil(size / n_shards)
            cumulative_size = offset + size

            for shard_id in range(1, n_shards + 1):
                step_size = min(shard_size, cumulative_size - offset)
                shard_path = self._get_shard_path(split, shard_id, step_size)
                # Generate a subset of indices to select only a subset of
                # audio-files/labels for the current shard.
                file_indices = np.arange(offset, offset + step_size)
                self._write_tfrecord_file(shard_path, file_indices)
                offset += step_size

        print('Number of training examples: {}'.format(self.n_train))
        print('Number of testing examples: {}'.format(self.n_test))
        print('Number of validation examples: {}'.format(self.n_val))
        print('TFRecord files saved to {}'.format(self.output_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--meta-data-csv', type=str, dest='meta_csv',
                        default=_DEFAULT_META_CSV,
                        help='File containing audio file-paths and '
                             'corresponding labels. (default: %(default)s)')

    parser.add_argument('-o', '--output-dir', type=str, dest='output_dir',
                        default=_DEFAULT_OUTPUT_DIR,
                        help='Output directory to store TFRecord files.'
                             '(default: %(default)s)')

    parser.add_argument('--num-shards-train', type=int,
                        dest='n_shards_train',
                        default=_DEFAULT_NUM_SHARDS_TRAIN,
                        help='Number of shards to divide training set '
                             'TFRecords into. (default: %(default)s)')

    parser.add_argument('--num-shards-test', type=int,
                        dest='n_shards_test',
                        default=_DEFAULT_NUM_SHARDS_TEST,
                        help='Number of shards to divide testing set '
                             'TFRecords into. (default: %(default)s)')

    parser.add_argument('--num-shards-val', type=int,
                        dest='n_shards_val',
                        default=_DEFAULT_NUM_SHARDS_VAL,
                        help='Number of shards to divide validation set '
                             'TFRecords into. (default: %(default)s)')

    parser.add_argument('--test-size', type=float,
                        dest='test_size',
                        default=_DEFAULT_TEST_SIZE,
                        help='Fraction of examples in the testing set. '
                             '(default: %(default)s)')

    parser.add_argument('--val-size', type=float,
                        dest='val_size',
                        default=_DEFAULT_VAL_SIZE,
                        help='Fraction of examples in the validation set. '
                             '(default: %(default)s)')

    return parser.parse_args()


def main(args):
    converter = TFRecordsConverter(args.meta_csv,
                                   args.output_dir,
                                   args.n_shards_train,
                                   args.n_shards_test,
                                   args.n_shards_val,
                                   args.test_size,
                                   args.val_size)
    converter.convert()


if __name__ == '__main__':
    main(parse_args())
