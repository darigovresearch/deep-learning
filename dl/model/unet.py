import os
import logging
import settings

from datetime import datetime
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.callbacks import *
from keras.layers import *

from keras import backend


class UNet:
    """
    Source: https://github.com/usnistgov/semantic-segmentation-unet/blob/master/UNet/model.py
    """
    def __init__(self, input_size, is_pretrained, is_saved):
        load_unet_parameters = settings.DL_PARAM['unet']

        self.learning_rate = load_unet_parameters['learning_rate']
        self.batch_size = load_unet_parameters['batch_size']
        self.num_filters = load_unet_parameters['filters']
        self.kernel_size = load_unet_parameters['kernel_size']
        self.deconv_kernel_size = load_unet_parameters['deconv_kernel_size']
        self.pooling_stride = load_unet_parameters['pooling_stride']
        self.dropout_rate = load_unet_parameters['dropout_rate']
        self.number_classes = len(load_unet_parameters['classes'])
        self.number_channels = load_unet_parameters['input_size_c']

        self.inputs = Input(shape=input_size)

        self.loss_fn = CategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        suffix = "model-input" + str(input_size) + "-batch" + str(load_unet_parameters['batch_size']) + "-drop" + \
                 str(load_unet_parameters['dropout_rate']).replace(".", "") + "-epoch" + "{epoch:02d}.hdf5"
        filepath = os.path.join(load_unet_parameters['output_checkpoints'], suffix)
        self.callbacks = [
            EarlyStopping(mode='max', monitor='loss', patience=20),
            ModelCheckpoint(filepath=filepath, monitor='accuracy', save_best_only=True,
                            save_weights_only='True', mode='max', verbose=1),
            TensorBoard(log_dir=load_unet_parameters['tensorboard_log_dir'], write_graph=True),
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
                https://stackoverflow.com/questions/53322488/implementing-u-net-for-multi-class-road-segmentation
        """
        logging.info(">>>> Settings up UNET model...")

        conv_1 = self.conv_block(self.inputs, n_filters=self.num_filters, size=self.kernel_size)
        conv1_out = self.pool(conv_1, pool_size=self.pooling_stride)
        conv_2 = self.conv_block(conv1_out, n_filters=(2 * self.num_filters), size=self.kernel_size)
        conv2_out = self.pool(conv_2, pool_size=self.pooling_stride)
        conv_3 = self.conv_block(conv2_out, n_filters=(4 * self.num_filters), size=self.kernel_size)
        conv3_out = self.pool(conv_3, pool_size=self.pooling_stride)
        conv_4 = self.conv_block(conv3_out, n_filters=(8 * self.num_filters), size=self.kernel_size)
        conv4_out = self.pool(conv_4, pool_size=self.pooling_stride)
        conv4_out = Dropout(rate=self.dropout_rate)(conv4_out)

        bottleneck = self.conv_block(conv4_out, n_filters=(16 * self.num_filters), size=self.kernel_size)
        bottleneck = Dropout(rate=self.dropout_rate)(bottleneck)

        deconv_4 = self.deconv_block(bottleneck, residual=conv_4, n_filters=(8 * self.num_filters),
                                     size=self.deconv_kernel_size, stride=self.pooling_stride)
        deconv_4 = Dropout(rate=self.dropout_rate)(deconv_4)
        deconv_5 = self.deconv_block(deconv_4, residual=conv_3, n_filters=(4 * self.num_filters),
                                     size=self.deconv_kernel_size, stride=self.pooling_stride)
        deconv_5 = Dropout(rate=self.dropout_rate)(deconv_5)
        deconv_6 = self.deconv_block(deconv_5, residual=conv_2, n_filters=(2 * self.num_filters),
                                     size=self.deconv_kernel_size, stride=self.pooling_stride)
        deconv_7 = self.deconv_block(deconv_6, residual=conv_1, n_filters=self.num_filters,
                                     size=self.deconv_kernel_size, stride=self.pooling_stride)

        output_layer = Conv2D(filters=self.number_classes, kernel_size=(1, 1))(deconv_7)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('softmax')(output_layer)

        model_obj = Model(self.inputs, output_layer, name='unet')
        model_obj.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

        logging.info(">>>> Done!")

        return model_obj

    def dice_coef(self, y_true, y_pred):
        return (2. * backend.sum(y_true * y_pred) + 1.) / (backend.sum(y_true) + backend.sum(y_pred) + 1.)

    def pool(self, tensor, pool_size):
        output = MaxPooling2D(pool_size=pool_size)(tensor)
        return output

    def conv_block(self, tensor, n_filters, size=3, padding='same', initializer="he_normal"):
        conv = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding,
                      kernel_initializer=initializer)(tensor)
        conv = BatchNormalization(axis=1)(conv)
        conv = Activation("relu")(conv)
        conv = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding,
                      kernel_initializer=initializer)(conv)
        conv = BatchNormalization(axis=1)(conv)
        conv = Activation("relu")(conv)
        return conv

    def deconv_block(self, tensor, n_filters, residual, size=3, stride=2, padding='same'):
        deconv = Conv2DTranspose(n_filters, kernel_size=(size, size), strides=(stride, stride), padding=padding)(tensor)
        deconv = concatenate([deconv, residual], axis=3)
        deconv = self.conv_block(deconv, n_filters=n_filters)
        return deconv

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
