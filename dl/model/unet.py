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

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
        pool_1 = tf.keras.layers.Dropout(rate=self.dropout_rate)(pool_1)

        conv_2 = self._conv_block(pool_1, 2 * self.num_filters, self.kernel_size)
        conv_2 = self._conv_block(conv_2, 2 * self.num_filters, self.kernel_size)
        pool_2 = self._pool(conv_2, self.pooling_stride)
        pool_2 = tf.keras.layers.Dropout(rate=self.dropout_rate)(pool_2)

        conv_3 = self._conv_block(pool_2, 4 * self.num_filters, self.kernel_size)
        conv_3 = self._conv_block(conv_3, 4 * self.num_filters, self.kernel_size)
        pool_3 = self._pool(conv_3, self.pooling_stride)
        pool_3 = tf.keras.layers.Dropout(rate=self.dropout_rate)(pool_3)

        conv_4 = self._conv_block(pool_3, 8 * self.num_filters, self.kernel_size)
        conv_4 = self._conv_block(conv_4, 8 * self.num_filters, self.kernel_size)
        pool_4 = self._pool(conv_4, self.pooling_stride)
        pool_4 = tf.keras.layers.Dropout(rate=self.dropout_rate)(pool_4)

        bottleneck = self._conv_block(pool_4, 16 * self.num_filters, self.kernel_size)

        deconv_4 = self._deconv_block(bottleneck, 8 * self.num_filters, self.deconv_kernel_size,
                                      stride=self.pooling_stride)
        deconv_4 = tf.keras.layers.Concatenate(axis=1)([deconv_4, conv_4])
        deconv_4 = tf.keras.layers.Dropout(rate=self.dropout_rate)(deconv_4)
        deconv_4 = self._conv_block(deconv_4, 8 * self.num_filters, self.kernel_size)
        deconv_4 = self._conv_block(deconv_4, 8 * self.num_filters, self.kernel_size)

        deconv_3 = self._deconv_block(deconv_4, 4 * self.num_filters, self.deconv_kernel_size,
                                      stride=self.pooling_stride)
        deconv_3 = tf.keras.layers.Concatenate(axis=1)([deconv_3, conv_3])
        deconv_3 = tf.keras.layers.Dropout(rate=self.dropout_rate)(deconv_3)
        deconv_3 = self._conv_block(deconv_3, 4 * self.num_filters, self.kernel_size)
        deconv_3 = self._conv_block(deconv_3, 4 * self.num_filters, self.kernel_size)

        deconv_2 = self._deconv_block(deconv_3, 2 * self.num_filters, self.deconv_kernel_size,
                                      stride=self.pooling_stride)
        deconv_2 = tf.keras.layers.Concatenate(axis=1)([deconv_2, conv_2])
        deconv_2 = tf.keras.layers.Dropout(rate=self.dropout_rate)(deconv_2)
        deconv_2 = self._conv_block(deconv_2, 2 * self.num_filters, self.kernel_size)
        deconv_2 = self._conv_block(deconv_2, 2 * self.num_filters, self.kernel_size)

        deconv_1 = self._deconv_block(deconv_2, self.num_filters, self.deconv_kernel_size,
                                      stride=self.pooling_stride)
        deconv_1 = tf.keras.layers.Concatenate(axis=1)([deconv_1, conv_1])
        deconv_1 = tf.keras.layers.Dropout(rate=self.dropout_rate)(deconv_1)
        deconv_1 = self._conv_block(deconv_1, self.num_filters, self.kernel_size)
        deconv_1 = self._conv_block(deconv_1, self.num_filters, self.kernel_size)

        outputs = Conv2D(self.number_classes, (1, 1), activation='softmax')(deconv_1)

        model_obj = tf.keras.Model(self.inputs, outputs, name='unet')
        model_obj.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

        logging.info(">>>> Done!")

        return model_obj

    def build_model_2(self):
        """
        Source: https://github.com/usnistgov/semantic-segmentation-unet
        """
        logging.info(">>>> Settings up UNET model...")

        conv1 = self._conv_block_2(self.inputs, nfilters=self.num_filters)
        conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = self._conv_block_2(conv1_out, nfilters=self.num_filters * 2)
        conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = self._conv_block_2(conv2_out, nfilters=self.num_filters * 4)
        conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = self._conv_block_2(conv3_out, nfilters=self.num_filters * 8)
        conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv4_out = Dropout(0.5)(conv4_out)
        conv5 = self._conv_block_2(conv4_out, nfilters=self.num_filters * 16)
        conv5 = Dropout(0.5)(conv5)

        deconv6 = self._deconv_block_2(conv5, residual=conv4, nfilters=self.num_filters * 8)
        deconv6 = Dropout(0.5)(deconv6)
        deconv7 = self._deconv_block_2(deconv6, residual=conv3, nfilters=self.num_filters * 4)
        deconv7 = Dropout(0.5)(deconv7)
        deconv8 = self._deconv_block_2(deconv7, residual=conv2, nfilters=self.num_filters * 2)
        deconv9 = self._deconv_block_2(deconv8, residual=conv1, nfilters=self.num_filters)

        output_layer = Conv2D(filters=self.number_classes, kernel_size=(1, 1))(deconv9)
        output_layer = BatchNormalization()(output_layer)

        # reshape = Reshape((self.img_rows * self.img_cols, 12), input_shape=(self.img_rows, self.img_cols, 12))(conv9)

        softmax = Activation('softmax')(output_layer)

        model_obj = tf.keras.Model(self.inputs, softmax, name='unet')
        model_obj.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

        logging.info(">>>> Done!")

        return model_obj

    def build_model_3(self):
        """
        Source: https://github.com/usnistgov/semantic-segmentation-unet
        """
        logging.info(">>>> Settings up UNET model...")

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(self.inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(filters=self.number_classes, kernel_size=(1, 1))(conv9)
        softmax = Activation('softmax')(conv10)

        model_obj = tf.keras.Model(self.inputs, softmax, name='unet')
        model_obj.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

        logging.info(">>>> Done!")

        return model_obj

    @staticmethod
    def _pool(tensor, nfilters):
        output = tf.keras.layers.MaxPooling2D(pool_size=nfilters)(tensor)
        return output

    @staticmethod
    def _conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
        output = Conv2D(filters=nfilters, kernel_size=size, strides=1, padding=padding, activation=relu,
                        kernel_initializer=initializer)(tensor)
        output = BatchNormalization(axis=1)(output)
        output = Activation("relu")(output)
        return output

    @staticmethod
    def _deconv_block(tensor, nfilters, size=3, padding='same', stride=1):
        output = Conv2DTranspose(filters=nfilters, kernel_size=size, strides=stride, activation=None,
                                 padding=padding)(tensor)
        # output = BatchNormalization(axis=1)(output)
        return output

    @staticmethod
    def _conv_block_2(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
        x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def _deconv_block_2(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2), initializer="he_normal"):
        y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
        y = concatenate([y, residual], axis=3)

        y = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        return y

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