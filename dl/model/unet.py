import os
import logging
import settings

from datetime import datetime
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard


class UNet:
    """
    Python class intended for building the model UNet, mainly using Keras framework

    Source:
        - https://github.com/usnistgov/semantic-segmentation-unet/blob/master/UNet/model.py
        - https://github.com/keras-team/keras-io/blob/master/examples/vision/oxford_pets_image_segmentation.py
    """
    def __init__(self, input_size, is_pretrained, is_saved):
        load_unet_parameters = settings.DL_PARAM['unet']

        self.learning_rate = load_unet_parameters['learning_rate']
        self.batch_size = load_unet_parameters['batch_size']
        self.num_filters = load_unet_parameters['filters']
        self.kernel_size = load_unet_parameters['kernel_size']
        self.deconv_kernel_size = load_unet_parameters['deconv_kernel_size']
        self.pooling_size = load_unet_parameters['pool_size']
        self.pooling_stride = load_unet_parameters['pooling_stride']
        self.dropout_rate = load_unet_parameters['dropout_rate']
        self.number_classes = len(load_unet_parameters['classes'])
        self.number_channels = load_unet_parameters['input_size_c']

        self.inputs = Input(shape=input_size)

        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        suffix = "model-input" + str(input_size[0]) + "-" + str(input_size[1]) + "-batch" + \
                 str(load_unet_parameters['batch_size']) + "-drop" + \
                 str(load_unet_parameters['dropout_rate']).replace(".", "") + "-epoch" + "{epoch:02d}.h5"
        filepath = os.path.join(load_unet_parameters['output_checkpoints'], suffix)

        self.callbacks = [
            EarlyStopping(mode='max', monitor='accuracy', patience=30),
            ModelCheckpoint(filepath=filepath, save_best_only=True, save_weights_only='True'),
            TensorBoard(log_dir=load_unet_parameters['tensorboard_log_dir'], write_graph=True),
        ]
        self.model = self.build_model()

        if is_pretrained is True:
            logging.info(">> Loading pretrained weights: {}...".format(load_unet_parameters['pretrained_weights']))
            pretrained_weights = os.path.join(load_unet_parameters['output_checkpoints'],
                                              load_unet_parameters['pretrained_weights'])
            self.model.load_weights(pretrained_weights)

        if is_saved is True:
            timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M")
            model_path = os.path.join(load_unet_parameters['save_model_dir'], "unet-" + timestamp + ".hdf5")

            logging.info(">>>>>> Model built. Saving model in {}...".format(model_path))
            self.model.save(model_path)

    def build_model(self):
        """
        According to the UNet deep architecture presented in https://lmb.informatik.uni-freiburg.de/
        Publications/2015/RFB15a/, the method returns the exact model based on input dimensions and other parameters
        set in settings.py

        Source:
            - https://github.com/usnistgov/semantic-segmentation-unet
            - https://stackoverflow.com/questions/53322488/implementing-u-net-for-multi-class-road-segmentation
        """
        logging.info(">>>> Settings up UNET model...")

        conv_1 = self.conv_block(self.inputs, n_filters=self.num_filters, size=self.kernel_size)
        conv1_out = self.pool(conv_1, pool_size=self.pooling_size, stride=self.pooling_stride)
        conv_2 = self.conv_block(conv1_out, n_filters=(2 * self.num_filters), size=self.kernel_size)
        conv2_out = self.pool(conv_2, pool_size=self.pooling_size, stride=self.pooling_stride)
        conv_3 = self.conv_block(conv2_out, n_filters=(4 * self.num_filters), size=self.kernel_size)
        conv3_out = self.pool(conv_3, pool_size=self.pooling_size, stride=self.pooling_stride)
        conv_4 = self.conv_block(conv3_out, n_filters=(8 * self.num_filters), size=self.kernel_size)
        conv4_out = self.pool(conv_4, pool_size=self.pooling_size, stride=self.pooling_stride)
        conv4_out = layers.Dropout(rate=self.dropout_rate)(conv4_out)

        bottleneck = self.conv_block(conv4_out, n_filters=(16 * self.num_filters), size=self.kernel_size)
        bottleneck = layers.Dropout(rate=self.dropout_rate)(bottleneck)

        deconv_4 = self.deconv_block(bottleneck, residual=conv_4, n_filters=(8 * self.num_filters),
                                     size=self.deconv_kernel_size, stride=self.pooling_stride)
        deconv_4 = layers.Dropout(rate=self.dropout_rate)(deconv_4)
        deconv_5 = self.deconv_block(deconv_4, residual=conv_3, n_filters=(4 * self.num_filters),
                                     size=self.deconv_kernel_size, stride=self.pooling_stride)
        deconv_5 = layers.Dropout(rate=self.dropout_rate)(deconv_5)
        deconv_6 = self.deconv_block(deconv_5, residual=conv_2, n_filters=(2 * self.num_filters),
                                     size=self.deconv_kernel_size, stride=self.pooling_stride)
        deconv_7 = self.deconv_block(deconv_6, residual=conv_1, n_filters=self.num_filters,
                                     size=self.deconv_kernel_size, stride=self.pooling_stride)

        output_layer = layers.Conv2D(filters=self.number_classes, kernel_size=(1, 1))(deconv_7)
        output_layer = layers.BatchNormalization()(output_layer)
        output_layer = layers.Activation('softmax')(output_layer)

        model_obj = Model(self.inputs, output_layer, name='unet')

        logging.info(">>>> Done!")

        return model_obj

    def pool(self, tensor, pool_size, stride):
        """
        Pooling layer setup. Build pooling layer according to the tensor and pool_size

        :param tensor: the input to the pooling
        :param pool_size: the dimension of the pooling
        :param stride: the stride
        :return output: the resultant tensor after pooling
        """
        output = layers.MaxPooling2D(pool_size=pool_size, strides=stride)(tensor)
        return output

    def conv_block(self, tensor, n_filters, size=3, padding='same', initializer="he_normal"):
        """
        Convolution layers setup. Build convolutional layer according to the tensor and params

        :param tensor: the input to the pooling
        :param n_filters: the exactly number of filter
        :param size: the squared kernel size. Default value is 3
        :param padding: type of padding. Default value is same
        :param initializer: type of the initializer. Default value is he_normal
        :return conv: the resultant tensor after two consecutive convolutions
        """
        conv = layers.Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding,
                             kernel_initializer=initializer)(tensor)
        conv = layers.BatchNormalization(axis=1)(conv)
        conv = layers.Activation("relu")(conv)
        conv = layers.Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding,
                             kernel_initializer=initializer)(conv)
        conv = layers.BatchNormalization(axis=1)(conv)
        conv = layers.Activation("relu")(conv)
        return conv

    def deconv_block(self, tensor, n_filters, residual, size=3, stride=2, padding='same'):
        """
        Deconvolution layer setup. Build deconvolutional layer according to the tensor and params

        :param tensor: the input to the pooling
        :param n_filters: the exactly number of filter
        :param residual:
        :param size: the squared kernel size. Default value is 3
        :param stride: type of the initializer. Default value is he_normal
        :param padding: type of padding. Default value is same
        :return deconv: the resultant tensor after deconvolution (transposition) operation
        """
        deconv = layers.Conv2DTranspose(n_filters, kernel_size=(size, size),
                                        strides=(stride, stride), padding=padding)(tensor)
        deconv = layers.concatenate([deconv, residual], axis=3)
        deconv = self.conv_block(deconv, n_filters=n_filters)
        return deconv

    def get_model(self):
        """
        :return model: the respective compiled model
        """
        return self.model

    def get_pooling_size(self):
        """
        :return pooling_size: the respective pooling size.
        """
        return self.optimizer

    def get_optimizer(self):
        """
        :return optimizer: the respective optimizer. Default is Adam
        """
        return self.optimizer

    def get_learning_rate(self):
        """
        :return learning_rate: the respective optimizer's learning rate. Previously defined in settings.py
        """
        return self.optimizer.learning_rate

    def get_batch_size(self):
        """
        :return batch_size: the respective model's batch_size. Previously defined in settings.py
        """
        return self.batch_size

    def get_num_channels(self):
        """
        :return number_channels: the number of samples's bands
        """
        return self.number_channels

    def get_num_classes(self):
        """
        :return number_classes: the respective number of classes. Previously defined in settings.py
        """
        return self.number_classes

    def get_inputs(self):
        """
        :return inputs: the input's dimension. Previously defined in settings.py
        """
        return self.inputs

    def get_loss(self):
        """
        :return loss_fn: the respective loss function
        """
        return self.loss_fn

    def get_callbacks(self):
        """
        :return callbacks: the model's callback
        """
        return self.callbacks
