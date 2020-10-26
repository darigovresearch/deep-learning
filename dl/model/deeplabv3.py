import os
import argparse
import time
import platform
import json
import warnings
import shutil
import random

import numpy as np
import cv2 as cv
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dropout
from tensorflow.keras.layers import Concatenate, Lambda, Activation, AveragePooling2D, SeparableConv2D
from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV2, Xception
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite

from ku.metrics_ext import MeanIoUExt
from ku.loss_ext import CategoricalCrossentropyWithLabelGT

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True
MLFLOW_USAGE = False
NUM_CLASSES = 21

MODE_TRAIN = 0
MODE_VAL = 1
MODE_TEST = 2

BASE_MODEL_MOBILENETV2 = 0
BASE_MODEL_XCEPTION = 1


class SemanticSegmentation(object):
    """Keras Semantic segmentation model of DeeplabV3+"""

    # Constants.
    # MODEL_PATH = 'semantic_segmentation_deeplabv3plus'
    MODEL_PATH = 'semantic_segmentation_deeplabv3plus.h5'
    TF_LITE_MODEL_PATH = 'semantic_segmentation_deeplabv3plus.tflite'

    # MODEL_PATH = 'semantic_segmentation_deeplabv3plus_is224_lr0_0001_ep344.h5'

    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dictionary
            Semantic segmentation model configuration dictionary.
        """

        # Check exception.
        assert conf['nn_arch']['output_stride'] == 8 or conf['nn_arch']['output_stride'] == 16

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.model_loading = self.conf['model_loading']

        if self.model_loading:
            opt = optimizers.Adam(lr=self.hps['lr']
                                  , beta_1=self.hps['beta_1']
                                  , beta_2=self.hps['beta_2']
                                  , decay=self.hps['decay'])
            with CustomObjectScope({'CategoricalCrossentropyWithLabelGT': CategoricalCrossentropyWithLabelGT,
                                    'MeanIoUExt': MeanIoUExt}):
                if self.conf['multi_gpu']:
                    self.model = load_model(os.path.join(self.raw_data_path, self.MODEL_PATH))

                    self.parallel_model = multi_gpu_model(self.model, gpus=self.conf['num_gpus'])
                    self.parallel_model.compile(optimizer=opt
                                                , loss=self.model.losses
                                                , metrics=self.model.metrics)
                else:
                    self.model = load_model(os.path.join(self.raw_data_path, self.MODEL_PATH))
                    # self.model.compile(optimizer=opt,
                    #           , loss=CategoricalCrossentropyWithLabelGT(num_classes=self.nn_arch['num_classes'])
                    #           , metrics=[MeanIoUExt(num_classes=NUM_CLASSES)]
        else:
            # Design the semantic segmentation model.
            # Load a base model.
            if self.conf['base_model'] == BASE_MODEL_MOBILENETV2:
                # Load mobilenetv2 as the base model.
                mv2 = MobileNetV2(include_top=False)  # , depth_multiplier=self.nn_arch['mv2_depth_multiplier'])

                if self.nn_arch['output_stride'] == 8:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer(
                        'block_5_add').output)  # Layer satisfying output stride of 8.
                else:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer(
                        'block_12_add').output)  # Layer satisfying output stride of 16.

                self.base.trainable = True
                for layer in self.base.layers: layer.trainable = True  # ?

                self.base._init_set_name('base')
            elif self.conf['base_model'] == BASE_MODEL_XCEPTION:
                # Load xception as the base model.
                mv2 = Xception(include_top=False)  # , depth_multiplier=self.nn_arch['mv2_depth_multiplier'])

                if self.nn_arch['output_stride'] == 8:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer(
                        'block4_sepconv2_bn').output)  # Layer satisfying output stride of 8.
                else:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer(
                        'block13_sepconv2_bn').output)  # Layer satisfying output stride of 16.

                self.base.trainable = True
                for layer in self.base.layers: layer.trainable = True  # ?

                self.base._init_set_name('base')

                # Make the encoder-decoder model.
            self._make_encoder()
            self._make_decoder()

            inputs = self.encoder.inputs
            features = self.encoder(inputs)
            outputs = self.decoder([inputs[0], features]) if self.nn_arch['boundary_refinement'] \
                else self.decoder(features)

            self.model = Model(inputs, outputs)

            # Compile.
            opt = optimizers.Adam(lr=self.hps['lr']
                                  , beta_1=self.hps['beta_1']
                                  , beta_2=self.hps['beta_2']
                                  , decay=self.hps['decay'])

            self.model.compile(optimizer=opt
                               , loss=CategoricalCrossentropyWithLabelGT(num_classes=self.nn_arch['num_classes'])
                               , metrics=[MeanIoUExt(num_classes=NUM_CLASSES)])
            self.model._init_set_name('deeplabv3plus_mnv2')

            if self.conf['multi_gpu']:
                self.parallel_model = multi_gpu_model(self.model, gpus=self.conf['num_gpus'])
                self.parallel_model.compile(optimizer=opt
                                            , loss=self.model.losses
                                            , metrics=self.model.metrics)

    def _make_encoder(self):
        """Make encoder."""
        assert hasattr(self, 'base')

        # Inputs.
        input_image = Input(shape=(self.nn_arch['image_size']
                                   , self.nn_arch['image_size']
                                   , 3)
                            , name='input_image')

        # Extract feature.
        x = self.base(input_image)

        # Conduct dilated convolution pooling.
        pooled_outputs = []
        for conf in self.nn_arch["encoder_middle_conf"]:
            if conf['input'] == -1:
                x2 = x  # ?
            else:
                x2 = pooled_outputs[conf['input']]

            if conf['op'] == 'conv':
                if conf['kernel'] == 1:
                    x2 = Conv2D(self.nn_arch['reduction_size']
                                , kernel_size=1
                                , padding='same'
                                , use_bias=False
                                , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x2)
                    x2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x2)
                    x2 = Activation('relu')(x2)
                else:
                    # Split separable conv2d.
                    x2 = SeparableConv2D(self.nn_arch['reduction_size']  # ?
                                         , conf['kernel']
                                         , depth_multiplier=1
                                         , dilation_rate=(conf['rate'][0] * self.nn_arch['conv_rate_multiplier']
                                                          , conf['rate'][1] * self.nn_arch['conv_rate_multiplier'])
                                         , padding='same'
                                         , use_bias=False
                                         , kernel_initializer=initializers.TruncatedNormal())(x2)
                    x2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x2)
                    x2 = Activation('relu')(x2)
                    x2 = Conv2D(self.nn_arch['reduction_size']
                                , kernel_size=1
                                , padding='same'
                                , use_bias=False
                                , kernel_initializer=initializers.TruncatedNormal()
                                , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x2)
                    x2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x2)
                    x2 = Activation('relu')(x2)
            elif conf['op'] == 'pyramid_pooling':
                x2 = AveragePooling2D(pool_size=conf['kernel'], padding='valid')(x2)
                x2 = Conv2D(self.nn_arch['reduction_size']
                            , kernel_size=1
                            , padding='same'
                            , use_bias=False
                            , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x2)
                x2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x2)
                x2 = Activation('relu')(x2)

                target_size = conf['target_size_factor']  # ?
                x2 = Lambda(lambda x: K.resize_images(x
                                                      , target_size[0]
                                                      , target_size[1]
                                                      , "channels_last"
                                                      , interpolation='bilinear'))(x2)  # ?
            else:
                raise ValueError('Invalid operation.')

            pooled_outputs.append(x2)

        # Concatenate pooled tensors.
        x3 = Concatenate(axis=-1)(pooled_outputs)
        x3 = Dropout(rate=self.nn_arch['dropout_rate'])(x3)
        x3 = Conv2D(self.nn_arch['concat_channels']
                    , kernel_size=1
                    , padding='same'
                    , use_bias=False
                    , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x3)
        x3 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x3)
        x3 = Activation('relu')(x3)
        # output = Dropout(rate=self.nn_arch['dropout_rate'])(x3)
        output = x3

        self.encoder = Model(input_image, output)
        self.encoder._init_set_name('encoder')

    def _make_decoder(self):
        """Make decoder."""
        assert hasattr(self, 'base') and hasattr(self, 'encoder')

        inputs = self.encoder.outputs
        features = Input(shape=K.int_shape(inputs[0])[1:])

        if self.nn_arch['boundary_refinement']:
            # Refine boundary.
            low_features = Input(shape=K.int_shape(self.encoder.inputs[0])[1:])
            x = self._refine_boundary(low_features, features)
        else:
            x = features

        # Upsampling & softmax.
        x = Conv2D(self.nn_arch['num_classes']
                   , kernel_size=3
                   , padding='same'
                   , use_bias=False
                   , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x)  # Kernel size?

        output_stride = self.nn_arch['output_stride']

        if self.nn_arch['boundary_refinement']:
            output_stride = output_stride / 8 if output_stride == 16 else output_stride / 4

        x = Lambda(lambda x: K.resize_images(x
                                             , output_stride
                                             , output_stride
                                             , "channels_last"
                                             , interpolation='bilinear'))(x)  # ?
        outputs = Activation('softmax')(x)

        self.decoder = Model(inputs=[low_features, features], outputs=outputs) if self.nn_arch['boundary_refinement'] \
            else Model(inputs=[features], outputs=outputs)
        self.decoder._init_set_name('decoder')

    def _refine_boundary(self, low_features, features):
        """Refine segmentation boundary.

        Parameters
        ----------
        low_features: Tensor
            Image input tensor.
        features: Tensor
            Encoder's output tensor.

        Returns
        -------
        Refined features.
            Tensor
        """
        low_features = self.base(low_features)
        low_features = Conv2D(48
                              , kernel_size=1
                              , padding='same'
                              , use_bias=False
                              , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(low_features)
        low_features = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(low_features)
        low_features = Activation('relu')(low_features)

        # Resize low_features, features.
        output_stride = self.nn_arch['output_stride']
        low_features = Lambda(lambda x: K.resize_images(x
                                                        , output_stride / 2
                                                        , output_stride / 2
                                                        , "channels_last"
                                                        , interpolation='bilinear'))(low_features)  # ?
        features = Lambda(lambda x: K.resize_images(x
                                                    , output_stride / 2
                                                    , output_stride / 2
                                                    , "channels_last"
                                                    , interpolation='bilinear'))(features)  # ?

        x = Concatenate(axis=-1)([low_features, features])

        return x