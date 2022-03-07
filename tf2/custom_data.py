from dataclasses import replace
from operator import index
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import pickle
from glob import glob
from absl import logging
import sys

from absl import flags
FLAGS = flags.FLAGS


def getBuilder(dataset, *args, **kwargs):
    if dataset == 'mvtech':
        return MVTechBuilder(dataset, *args, **kwargs)
    elif dataset == 'bmw':
        return BMWBuilder(dataset, *args, **kwargs)
    else:
        return tfds.builder(dataset, *args, **kwargs)


class StandardBuilder():

    def __init__(self, *args, train_mode='train_then_eval',
                #  use_all_data=True, 
                 anomaly_perc=0.8,
                 test_perc=0.2, 
                 **kwargs):
        self.train_mode = train_mode
        # self.use_all_data = use_all_data
        self.anomaly_perc = anomaly_perc
        self.test_perc = test_perc
        self._info = None
        

    def split_data_set(self, data_frame, neg_mask, pos_mask):
      
        n_neg_total = data_frame[neg_mask].shape[0]
        n_pos_total = data_frame[pos_mask].shape[0]

        # compute the number of positive and negative samples we need to ensure anomaly_perc
        n_pos = round((1/(1-self.anomaly_perc) -1) * n_neg_total)
        n_neg = round((1/self.anomaly_perc - 1) * n_pos_total)

        # check if there are enough positive samples to ensure anomaly_perc if we use all neg samples
        # if there is not, use all pos sample and instead drop some neg samples 
        if n_pos <= n_pos_total:
            n_pos_train = round(n_pos * (1 - self.test_perc))
            n_neg_train = round(n_neg_total * (1 - self.test_perc))
            # let's remove the pos samples we don't need
            # so that that test set will also be balanced
            pos_frame = data_frame[pos_mask]
            drop_idx = pos_frame.sample(n=n_pos_total - n_pos, replace=False, axis=0).index
            balanced_df = data_frame.drop(index=drop_idx)
        elif n_neg <= n_neg_total: 
            # if we don't have enough pos samples let's drop some neg samples then
            n_neg_train = round(n_neg * (1 - self.test_perc))
            n_pos_train = round(n_pos_total * (1 - self.test_perc))
            # let's remove the neg samples we don't need
            # so that that test set will also be balanced
            neg_frame = data_frame[neg_mask]
            drop_idx = neg_frame.sample(n=n_neg_total - n_neg, replace=False, axis=0).index
            balanced_df = data_frame.drop(index=drop_idx)
        else:
            raise Exception('Error when computing anomaly_perc split')

        """
        if self.anomaly_perc < self.test_perc:
            # include more IOs
            self.test_perc = self.test_perc + \
                                    (self.test_perc - self.anomaly_perc)
        """


        neg_incl = balanced_df[balanced_df.lbl == 'IO']
        pos_incl = balanced_df[balanced_df.lbl == 'NIO']
        neg_incl = neg_incl.sample(n=n_neg_train, replace=False, axis=0)
        pos_incl = pos_incl.sample(n=n_pos_train, replace=False, axis=0)

        train_df = pd.concat([neg_incl, pos_incl])
        test_df = balanced_df.drop(index=train_df.index)

        logging.info('total images %d', balanced_df.shape[0])

        if FLAGS.show_debug:
            log_train = train_df.sample(n=5, replace=False)
            logging.info('train df {}', log_train)
            log_test = test_df.sample(n=5, replace=False)
            logging.info('test df {}', log_test)

       
        return (train_df, test_df)

    def prepare_dataset(self, train_df, test_df):
        logging.info('train images %d', train_df.shape[0])
        logging.info('test images %d', test_df.shape[0])

        train_ds = tf.data.Dataset.from_tensor_slices(list(zip(train_df.index.values, train_df.lbl)))
        test_ds = tf.data.Dataset.from_tensor_slices(list(zip(test_df.index.values, test_df.lbl)))

        info = Map({
            'splits': Map({
                'train': Map({
                    'num_examples': train_df.shape[0]
                }),
                'test': Map({
                    'num_examples': test_df.shape[0]
                })
            }),
            'features': Map({
                'label': Map({
                    'num_classes': 2
                })
            })
        })

        return {
            'info': info,
            'train_ds': train_ds,
            'test_ds': test_ds
        }


class MVTechBuilder(StandardBuilder):
    """
    This pretends do be a builder.
    `DatasetBuilder` has 3 key methods:
    * `DatasetBuilder.info`: documents the dataset, including feature
        names, types, and shapes, version, splits, citation, etc.
    * `DatasetBuilder.download_and_prepare`: downloads the source data
        and writes it to disk.
    * `DatasetBuilder.as_dataset`: builds an input pipeline using
        `tf.data.Dataset`s.
    """

    def __init__(self, dataset, data_dir, *args, **kwargs):
        super().__init__(**kwargs)
        print(kwargs)
        self.dataset = dataset
        if kwargs["categories"] != None:
            self.path = [os.path.join(data_dir, cat) for cat in kwargs["categories"]]
        else:
            self.path = [os.path.join(data_dir, '*')]

    def download_and_prepare(self):
        self._load_mvtech_dataset()

    @property
    def info(self):
        if self._info == None:
            raise ValueError('info is None. Call download_and_prepare() first.')
        return self._info

    def as_dataset(self, split=None, batch_size=None, shuffle_files=None, as_supervised=False, read_config=None):

        AUTOTUNE = tf.data.AUTOTUNE

        def get_label(status):
            # Convert the path to a list of path components
            # parts = tf.strings.split(file_path, os.path.sep)
            # The second to last is the class-directory
            l = status != 'IO'
            # Integer encode the label
            return int(l)  # tf.argmax(one_hot)

        def decode_img(img):
            # Convert the compressed string to a 3D uint8 tensor
            img = tf.io.decode_png(img, channels=3)
            # Resize the image to the desired size for testing
            # this is not needed because it's already done in build input func
            # img = tf.image.resize(img, [64, 64])
            return img

        def process(tpl):
            label = get_label(tpl[1])
            # Load the raw data from the file as a string
            img = tf.io.read_file(tpl[0])
            img = decode_img(img)
            return img, label
        
        if split == 'train':
            dataset = self.train_ds
        elif split == 'test':
            dataset = self.test_ds
        else:
            raise ValueError('Splits needs to be either train or test.')

        
        return dataset.map(process, num_parallel_calls=AUTOTUNE)

    def _load_mvtech_dataset(self):
        neg_files = []
        pos_files = []
        for path_cat in self.path:
            neg_files += glob(os.path.join(path_cat, 'train', 'good', '*.png'))
            neg_files += glob(os.path.join(path_cat, 'test', 'good', '*.png'))
            pos_files += glob(os.path.join(path_cat, 'test', '*', '*.png'))
            pos_files = [p for p in pos_files if 'good' not in p] # exclude all anomalies

        neg_df = pd.DataFrame(data={'lbl': ['IO'] * len(neg_files)}, index=neg_files)
        pos_df = pd.DataFrame(data={'lbl': ['NIO'] * len(pos_files)}, index=pos_files)

        df = pd.concat([neg_df, pos_df])
        neg_mask = df.lbl.values == 'IO'
        pos_mask = df.lbl.values == 'NIO'

        train_df, test_df = self.split_data_set(df, neg_mask, pos_mask)

        res = self.prepare_dataset(train_df, test_df)
        self.train_ds = res['train_ds']
        self.test_ds = res['test_ds']
        self._info = res['info']


class BMWBuilder(StandardBuilder):
    """
    This pretends do be a builder.
    `DatasetBuilder` has 3 key methods:
    * `DatasetBuilder.info`: documents the dataset, including feature
        names, types, and shapes, version, splits, citation, etc.
    * `DatasetBuilder.download_and_prepare`: downloads the source data
        and writes it to disk.
    * `DatasetBuilder.as_dataset`: builds an input pipeline using
        `tf.data.Dataset`s.
    """

    def __init__(self, dataset, data_dir,
                 load_existing_split=False,
                 results_dir=None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.path = data_dir
        self.load_existing_split=load_existing_split
        self.results_dir=results_dir

        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

    def download_and_prepare(self):
        self._load_bmw_dataset()

    @property
    def info(self):
        if self._info is None:
            raise ValueError('info is None. Call download_and_prepare() first.')
        return self._info

    def as_dataset(self, split=None, batch_size=None, shuffle_files=None, as_supervised=False, read_config=None):

        if tf.version.VERSION != '2.7.0':
            AUTOTUNE = tf.data.experimental.AUTOTUNE
        else:
            AUTOTUNE = tf.data.AUTOTUNE

        def get_label(status):
            l = status != 'IO'
            # Integer encode the label
            return int(l)  # tf.argmax(one_hot)

        def decode_img(img):
            # Convert the compressed string to a 3D uint8 tensor
            # img = tf.io.decode_png(img, channels=3)
            img = tf.io.decode_jpeg(img, channels=3)
            # Resize the image to the desired size for testing
            # this is not needed because it's already done in build input func
            # img = tf.image.resize(img, [64, 64])
            return img

        def process(tpl):
            label = get_label(tpl[1])
            # Load the raw data from the file as a string
            img = tf.io.read_file(tpl[0])
            img = decode_img(img)
            return img, label

        if split == 'train':
            dataset = self.train_ds
        elif split == 'test':
            dataset = self.test_ds
        else:
            raise ValueError('Splits needs to be either train or test.')

        return dataset.map(process, num_parallel_calls=AUTOTUNE)

    def _load_bmw_dataset(self):
        if not self.load_existing_split:
            annotations = pd.read_csv(os.path.join(self.path, 'annotation.csv'), index_col='file_name')
            neg_mask = annotations.lbl.values == 'IO'
            pos_mask = annotations.lbl.values != 'IO'

            train_df, test_df = self.split_data_set(annotations, neg_mask, pos_mask)

            if self.train_mode == 'finetune':
                if os.path.isfile(os.path.join(self.results_dir, "split.pkl")):
                    logging.warn("finetune mode and existing split detected. Change your run_id! Stopping")
                    sys.exit("Change our run_id!")

            with open(os.path.join(self.results_dir, "split.pkl"), "wb") as f:
                pickle.dump((train_df, test_df), f)
        else:
            if self.train_mode != 'finetune':
                logging.warn("finetune mode detected. existing split will be loaded. make sure this is what you want!")
            logging.info("loading existing split from {}".format(os.path.join(self.results_dir, "split.pkl")))
            with open(os.path.join(self.results_dir, "split.pkl"), "rb") as f:
                (train_df, test_df) = pickle.load(f)

        res = self.prepare_dataset(train_df, test_df)
        self.train_ds = res['train_ds']
        self.test_ds = res['test_ds']
        self._info = res['info']


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
