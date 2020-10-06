import logging
import settings

from keras.models import *
from keras.layers import *
from keras.optimizers import *


class UNet:
    def __init__(self):
        pass

    def model(self, input_size):
        """
        Source: https://github.com/zhixuhao/unet
        """
        load_unet_parameters = settings.DL_PARAM['unet']

        if input_size is not None:
            logging.info(">>>> Settings up UNET model...")

            inputs = Input(shape=input_size)

            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
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
            conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

            model_obj = Model(inputs, conv10)
            model_obj.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

            if load_unet_parameters['pretrained_weights'] != '':
                logging.info(">> Loading pretrained weights: {}...".format(load_unet_parameters['pretrained_weights']))
                model_obj.load_weights(load_unet_parameters['pretrained_weights'])

            logging.info(">>>> Done!")
        else:
            logging.warning(">>>> Input size is None. Model could not be retrieved")

        return model_obj

    def conv_block(self, tensor, nfilters, size=3, padding='same', initializer="he_normal"):
        x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def deconv_block(self, tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
        y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
        y = concatenate([y, residual], axis=3)
        y = self.conv_block(y, nfilters)
        return y

    def model_2(self, input_size):
        """
        Source: https://stackoverflow.com/questions/53322488/implementing-u-net-for-multi-class-road-segmentation
        """
        load_unet_parameters = settings.DL_PARAM['unet']

        if input_size is not None:
            logging.info(">>>> Settings up UNET model...")

            filters = 64
            nclasses = len(load_unet_parameters['classes'])

            input_layer = Input(shape=input_size, name='image_input')
            conv1 = self.conv_block(input_layer, nfilters=filters)
            conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = self.conv_block(conv1_out, nfilters=filters * 2)
            conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = self.conv_block(conv2_out, nfilters=filters * 4)
            conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
            conv4 = self.conv_block(conv3_out, nfilters=filters * 8)
            conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
            conv4_out = Dropout(0.5)(conv4_out)
            conv5 = self.conv_block(conv4_out, nfilters=filters * 16)
            conv5 = Dropout(0.5)(conv5)

            deconv6 = self.deconv_block(conv5, residual=conv4, nfilters=filters * 8)
            deconv6 = Dropout(0.5)(deconv6)
            deconv7 = self.deconv_block(deconv6, residual=conv3, nfilters=filters * 4)
            deconv7 = Dropout(0.5)(deconv7)
            deconv8 = self.deconv_block(deconv7, residual=conv2, nfilters=filters * 2)
            deconv9 = self.deconv_block(deconv8, residual=conv1, nfilters=filters)

            output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
            output_layer = BatchNormalization()(output_layer)
            output_layer = Activation('softmax')(output_layer)

            model_obj = Model(inputs=input_layer, outputs=output_layer, name='unet')
            model_obj.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

            if load_unet_parameters['pretrained_weights'] != '':
                logging.info(">> Loading pretrained weights: {}...".format(load_unet_parameters['pretrained_weights']))
                model_obj.load_weights(load_unet_parameters['pretrained_weights'])

            logging.info(">>>> Done!")
        else:
            logging.warning(">>>> Input size is None. Model could not be retrieved")

        return model_obj




