import os
import logging
import tensorflow as tf

from datetime import datetime
from satellite import settings
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


class DeepLabV3:
    """
    Keras Semantic segmentation model of DeeplabV3+

    Source:
        - https://keras.io/examples/vision/deeplabv3_plus/
    """

    def __init__(self, input_size, is_pretrained, is_saved):
        load_dlv3_parameters = settings.DL_PARAM['deeplabv3+']

        self.learning_rate = load_dlv3_parameters['learning_rate']
        self.num_filters = load_dlv3_parameters['filters']
        self.kernel_size = load_dlv3_parameters['kernel_size']
        self.deconv_kernel_size = load_dlv3_parameters['deconv_kernel_size']
        self.pooling_size = load_dlv3_parameters['pooling_size']
        self.pooling_stride = load_dlv3_parameters['pooling_stride']
        self.dropout_rate = load_dlv3_parameters['dropout_rate']
        self.number_classes = len(load_dlv3_parameters['classes'])
        self.width = load_dlv3_parameters['input_size_w']
        self.height = load_dlv3_parameters['input_size_h']
        self.number_channels = load_dlv3_parameters['input_size_c']

        self.inputs = keras.Input(shape=input_size)

        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        suffix = "model-input" + str(input_size[0]) + "-" + str(input_size[1]) + "-drop" + \
                 str(self.dropout_rate).replace(".", "") + \
                 "-epoch" + "{epoch:02d}.hdf5"
        filepath = os.path.join(load_dlv3_parameters['output_checkpoints'], suffix)

        self.callbacks = [
            EarlyStopping(mode='max', monitor='accuracy', patience=20),
            ModelCheckpoint(filepath=filepath, monitor='accuracy', save_best_only=True,
                            save_weights_only=True, mode='max', verbose=1),
            TensorBoard(log_dir=load_dlv3_parameters['tensorboard_log_dir'], write_graph=True)
        ]
        self.model = self.build_model()

        if is_pretrained is True:
            logging.info(">> Loading pretrained weights: {}...".format(load_dlv3_parameters['pretrained_weights']))
            pretrained_weights = os.path.join(load_dlv3_parameters['output_checkpoints'],
                                              load_dlv3_parameters['pretrained_weights'])
            self.model.load_weights(pretrained_weights)

        if is_saved is True:
            timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M")
            model_path = os.path.join(load_dlv3_parameters['save_model_dir'], "unet-" + timestamp + ".h5")

            logging.info(">>>>>> Model built. Saving model in {}...".format(model_path))
            self.model.save(model_path)

    def convolution_block(self, block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False):
        """
        """
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
        )(block_input)
        x = layers.BatchNormalization()(x)
        return tf.nn.relu(x)

    def DilatedSpatialPyramidPooling(self, dspp_input):
        """
        """
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = self.convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
                                       interpolation="bilinear",)(x)

        out_1 = self.convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = self.convolution_block(x, kernel_size=1)
        return output

    def build_model(self):
        """
        """
        resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=self.inputs)
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = self.DilatedSpatialPyramidPooling(x)

        input_a = layers.UpSampling2D(size=(self.width // 4 // x.shape[1], self.height // 4 // x.shape[2]),
                                      interpolation="bilinear",)(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x)
        x = self.convolution_block(x)
        x = layers.UpSampling2D(size=(self.width // x.shape[1], self.height // x.shape[2]),
                                interpolation="bilinear",)(x)
        model_output = layers.Conv2D(self.number_classes, kernel_size=(1, 1), padding="same")(x)
        return keras.Model(inputs=self.inputs, outputs=model_output)
