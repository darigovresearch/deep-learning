import os
import logging
import settings
import tensorflow as tf

from datetime import datetime
from keras.layers import *
from keras.activations import *


class UNet:
    """
    Source: https://github.com/usnistgov/semantic-segmentation-unet/blob/master/UNet/model.py
    """
    SIZE_FACTOR = 16
    RADIUS = 96

    def __init__(self, input_size, number_classes, number_channels, is_pretrained, is_saved):
        load_unet_parameters = settings.DL_PARAM['unet']

        self.learning_rate = load_unet_parameters['learning_rate']
        self.batch_size = load_unet_parameters['batch_size']
        self.num_filters = load_unet_parameters['filters']
        self.kernel_size = load_unet_parameters['kernel_size']
        self.deconv_kernel_size = load_unet_parameters['deconv_kernel_size']
        self.pooling_stride = load_unet_parameters['pooling_stride']
        self.dropout_rate = load_unet_parameters['dropout_rate']

        self.number_channels = number_channels
        self.number_classes = number_classes

        self.inputs = Input(shape=input_size, name='image_input')

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        filepath = os.path.join(load_unet_parameters['output_checkpoints'], "model-{epoch:02d}.hdf5")
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_accuracy',
                                               verbose=1, save_best_only=False, mode='auto'),
            tf.keras.callbacks.TensorBoard(log_dir=load_unet_parameters['tensorboard_log_dir']),
        ]
        self.model = self.build_model()

        if is_pretrained is True:
            logging.info(">> Loading pretrained weights: {}...".format(load_unet_parameters['pretrained_weights']))
            pretrained_weights = os.path.join(load_unet_parameters['output_checkpoints'],
                                              load_unet_parameters['pretrained_weights'])
            self.model.load_weights(pretrained_weights)

        if is_saved is True:
            logging.info(">>>>>> Model built. Saving model in {}...".format(load_unet_parameters['save_model_dir']))
            timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M)")
            self.model.save(os.path.join(load_unet_parameters['save_model_dir'], "unet-" + timestamp + ".h5"))

    def build_model(self):
        """
        Source: https://github.com/usnistgov/semantic-segmentation-unet
        """
        logging.info(">>>> Settings up UNET model...")

        conv_1 = self._conv_block(self.inputs, self.num_filters, self.kernel_size)
        conv_1 = self._conv_block(conv_1, self.num_filters, self.kernel_size)
        pool_1 = self._pool(conv_1, self.pooling_stride)
        conv_2 = self._conv_block(pool_1, 2 * self.num_filters, self.kernel_size)
        conv_2 = self._conv_block(conv_2, 2 * self.num_filters, self.kernel_size)
        pool_2 = self._pool(conv_2, self.pooling_stride)
        conv_3 = self._conv_block(pool_2, 4 * self.num_filters, self.kernel_size)
        conv_3 = self._conv_block(conv_3, 4 * self.num_filters, self.kernel_size)
        pool_3 = self._pool(conv_3, self.pooling_stride)
        conv_4 = self._conv_block(pool_3, 8 * self.num_filters, self.kernel_size)
        conv_4 = self._conv_block(conv_4, 8 * self.num_filters, self.kernel_size)
        conv_4 = tf.keras.layers.Dropout(rate=self.dropout_rate)(conv_4)
        pool_4 = self._pool(conv_4, self.pooling_stride)

        bottleneck = self._conv_block(pool_4, 16 * self.num_filters, self.kernel_size)
        bottleneck = self._conv_block(bottleneck, 16 * self.num_filters, self.kernel_size)
        bottleneck = tf.keras.layers.Dropout(rate=self.dropout_rate)(bottleneck)

        deconv_4 = self._deconv_block(bottleneck, 8 * self.num_filters, self.deconv_kernel_size,
                                      stride=self.pooling_stride)
        deconv_4 = tf.keras.layers.Concatenate(axis=1)([conv_4, deconv_4])
        deconv_4 = self._conv_block(deconv_4, 8 * self.num_filters, self.kernel_size)
        deconv_4 = self._conv_block(deconv_4, 8 * self.num_filters, self.kernel_size)

        deconv_3 = self._deconv_block(deconv_4, 4 * self.num_filters, self.deconv_kernel_size,
                                      stride=self.pooling_stride)
        deconv_3 = tf.keras.layers.Concatenate(axis=1)([conv_3, deconv_3])
        deconv_3 = self._conv_block(deconv_3, 4 * self.num_filters, self.kernel_size)
        deconv_3 = self._conv_block(deconv_3, 4 * self.num_filters, self.kernel_size)

        deconv_2 = self._deconv_block(deconv_3, 2 * self.num_filters, self.deconv_kernel_size,
                                      stride=self.pooling_stride)
        deconv_2 = tf.keras.layers.Concatenate(axis=1)([conv_2, deconv_2])
        deconv_2 = self._conv_block(deconv_2, 2 * self.num_filters, self.kernel_size)
        deconv_2 = self._conv_block(deconv_2, 2 * self.num_filters, self.kernel_size)

        deconv_1 = self._deconv_block(deconv_2, self.num_filters, self.deconv_kernel_size,
                                      stride=self.pooling_stride)
        deconv_1 = tf.keras.layers.Concatenate(axis=1)([conv_1, deconv_1])
        deconv_1 = self._conv_block(deconv_1, self.num_filters, self.kernel_size)
        deconv_1 = self._conv_block(deconv_1, self.num_filters, self.kernel_size)

        # convert NCHW to NHWC so that softmax axis is the last dimension
        # logits is [NHWC]
        logits = self._conv_block(deconv_1, self.number_classes, 1)
        # logits = tf.keras.layers.Permute((2, 3, 1))(logits)
        softmax = tf.keras.layers.Softmax(axis=-1, name='softmax')(logits)

        # softmax = Activation('softmax')(logits)

        model_obj = tf.keras.Model(self.inputs, softmax, name='unet')
        model_obj.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

        logging.info(">>>> Done!")

        return model_obj

    @staticmethod
    def _pool(tensor, nfilters):
        # , data_format = 'channels_first'
        output = tf.keras.layers.MaxPooling2D(pool_size=nfilters)(tensor)
        return output

    @staticmethod
    def _conv_block(tensor, nfilters, size=3, padding='same'):
        output = Conv2D(filters=nfilters, kernel_size=size, strides=1, padding=padding, activation=relu)(tensor)
        output = BatchNormalization(axis=1)(output)
        return output

    @staticmethod
    def _deconv_block(tensor, nfilters, size=3, padding='same', stride=1):
        output = Conv2DTranspose(filters=nfilters, kernel_size=size, strides=stride, activation=None,
                                 padding=padding)(tensor)
        output = BatchNormalization(axis=1)(output)
        return output

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_learning_rate(self):
        return self.optimizer.learning_rate

    def get_batch_size(self):
        return self.batch_size

    def get_num_channels(self):
        return self.number_channels

    def get_num_classes(self):
        return self.number_classes

    def get_inputs(self):
        return self.inputs

    def get_loss(self):
        return self.loss_fn

    def get_callbacks(self):
        return self.callbacks

    # def _conv_block(self, tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    #     x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    #     x = BatchNormalization()(x)
    #     x = Activation("relu")(x)
    #     x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    #     x = BatchNormalization()(x)
    #     x = Activation("relu")(x)
    #     return x
    #
    # def _deconv_block(self, tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    #     y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    #     y = concatenate([y, residual], axis=3)
    #     y = self._conv_block(y, nfilters)
    #     return y
    #
    # def model(self, input_size):
    #     """
    #     Source: https://stackoverflow.com/questions/53322488/implementing-u-net-for-multi-class-road-segmentation
    #     """
    #     load_unet_parameters = settings.DL_PARAM['unet']
    #
    #     if input_size is not None:
    #         logging.info(">>>> Settings up UNET model...")
    #
    #         filters = 64
    #         nclasses = len(load_unet_parameters['classes'])
    #
    #         input_layer = Input(shape=input_size, name='image_input')
    #         conv1 = self._conv_block(input_layer, nfilters=filters)
    #         conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)

    #         conv2 = self._conv_block(conv1_out, nfilters=filters * 2)
    #         conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    #         conv3 = self._conv_block(conv2_out, nfilters=filters * 4)
    #         conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    #         conv4 = self._conv_block(conv3_out, nfilters=filters * 8)
    #         conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    #         conv4_out = Dropout(0.5)(conv4_out)
    #         conv5 = self._conv_block(conv4_out, nfilters=filters * 16)
    #         conv5 = Dropout(0.5)(conv5)
    #
    #         deconv6 = self._deconv_block(conv5, residual=conv4, nfilters=filters * 8)
    #         deconv6 = Dropout(0.5)(deconv6)
    #         deconv7 = self._deconv_block(deconv6, residual=conv3, nfilters=filters * 4)
    #         deconv7 = Dropout(0.5)(deconv7)
    #         deconv8 = self._deconv_block(deconv7, residual=conv2, nfilters=filters * 2)
    #         deconv9 = self._deconv_block(deconv8, residual=conv1, nfilters=filters)
    #
    #         output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    #         output_layer = BatchNormalization()(output_layer)
    #         output_layer = Activation('softmax')(output_layer)
    #
    #         model_obj = Model(inputs=input_layer, outputs=output_layer, name='unet')
    #         model_obj.compile(optimizer=Adam(lr=1e-4),
    #                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                           metrics=['accuracy'])
    #
    #         if load_unet_parameters['pretrained_weights'] != '':
    #             logging.info(">> Loading pretrained weights: {}...".format(load_unet_parameters['pretrained_weights']))
    #             pretrained_weights = os.path.join(load_unet_parameters['output_checkpoints'],
    #                                               load_unet_parameters['pretrained_weights'])
    #             model_obj.load_weights(pretrained_weights)
    #         else:
    #             logging.info(">>>>>> Model built. Saving model in {}...".format(load_unet_parameters['save_model_dir']))
    #             timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M)")
    #             model_obj.save(os.path.join(load_unet_parameters['save_model_dir'],
    #                                         "unet-" + timestamp + ".h5"))
    #
    #         logging.info(">>>> Done!")
    #     else:
    #         logging.warning(">>>> Input size is None. Model could not be retrieved")
    #
    #     return model_obj


