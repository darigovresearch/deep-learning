import logging

from keras.models import *
from keras.layers import *
from keras.optimizers import *


class DeepLabV3plus:
    """
    Source: https://github.com/aruns2120/Semantic-Segmentation-Severstal/blob/
    master/DeepLab%20V3%2B/SeverstalSteel_DeepLabV3%2B.ipynb
    """
    def __init__(self):
        pass

    def SeparableConv_BN(self, filters, prefix='', stride=1, kernel_size=3, rate=1, depth_activation=False):
        stride = stride
        depth_activation = depth_activation
        # manual padding size when stride!=1
        if stride != 1:
            # effective kernel size = kernel_size + (kernel_size - 1) * (rate - 1)
            n_pads = (kernel_size + (kernel_size - 1) * (rate - 1) - 1) // 2
            zeropad = ZeroPadding2D(padding=n_pads)

        depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, dilation_rate=rate,
                                                     padding='same' if stride == 1 else 'valid',
                                                     name=prefix + '_depthW')
        batchnorm_d = BatchNormalization(name=prefix + '_depthW_BN')

        pointwise_conv = Conv2D(filters, kernel_size=1, padding='same', name=prefix + '_pointW')
        batchnorm_p = BatchNormalization(name=prefix + '_pointW_BN')

    def Xception_Block(self, depth_list, prefix='', residual_type=None, stride=1, rate=1, depth_activation=False,
                       return_skip=False):
        sepConv1 = self.SeparableConv_BN(filters=depth_list[0], prefix=prefix + '_sepConv1', stride=1, rate=rate,
                                         depth_activation=depth_activation)
        sepConv2 = self.SeparableConv_BN(filters=depth_list[1], prefix=prefix + '_sepConv2', stride=1, rate=rate,
                                         depth_activation=depth_activation)
        sepConv3 = self.SeparableConv_BN(filters=depth_list[2], prefix=prefix + '_sepConv3', stride=stride, rate=rate,
                                         depth_activation=depth_activation)

        if residual_type == 'conv':
            conv2D = self.Conv2D_custom(depth_list[2], prefix=prefix + '_conv_residual', stride=stride, kernel_size=1,
                                        rate=1)
            batchnorm_res = BatchNormalization(name=prefix + '_BN_residual')

        return_skip = return_skip
        residual_type = residual_type

    def call(self, x):
        output = self.sepConv1(x)
        output = self.sepConv2(output)
        skip = output  # skip connection to decoder
        output = self.sepConv3(output)

        if self.residual_type == 'conv':
            res = self.conv2D(x)
            res = self.batchnorm_res(res)
            output += res
        elif self.residual_type == 'sum':
            output += x
        else:
            if (self.residual_type):
                raise ValueError('Arg residual_type should be one of {conv, sum}')

        if self.return_skip:
            return output, skip

        return output

    def model(self, input_size, n_classes):

        input_size = input_size

        # Encoder block
        conv2d1 = Conv2D(32, (3, 3), strides=2, name='entry_conv1', padding='same')
        bn1 = BatchNormalization(name='entry_BN')
        custom_conv1 = self.Conv2D_custom(64, kernel_size=3, stride=1, prefix='entry_conv2')
        bn2 = BatchNormalization(name='conv2_s1_BN')

        entry_xception1 = self.Xception_Block([128, 128, 128], prefix='entry_x1', residual_type='conv', stride=2, rate=1)
        entry_xception2 = self.Xception_Block([256, 256, 256], prefix='entry_x2', residual_type='conv', stride=2, rate=1, return_skip=True)
        entry_xception3 = self.Xception_Block([728, 728, 728], prefix='entry_x3', residual_type='conv', stride=2, rate=1)

        middle_xception = [
            self.Xception_Block([728, 728, 728], prefix=f'middle_x{i + 1}', residual_type='sum', stride=1, rate=1) for i in
            range(16)]

        exit_xception1 = self.Xception_Block([728, 1024, 1024], prefix='exit_x1', residual_type='conv', stride=1, rate=1)
        exit_xception2 = self.Xception_Block([1536, 1536, 2048], prefix='exit_x2', residual_type=None, stride=1, rate=2, depth_activation=True)

        conv_feat = Conv2D(256, (1, 1), padding='same', name='conv_featureProj')
        bn_feat = BatchNormalization(name='featureProj_BN')
        atrous_conv1 = self.SeparableConv_BN(filters=256, prefix='aspp1', stride=1, rate=6, depth_activation=True)
        atrous_conv2 = self.SeparableConv_BN(filters=256, prefix='aspp2', stride=1, rate=12, depth_activation=True)
        atrous_conv3 = self.SeparableConv_BN(filters=256, prefix='aspp3', stride=1, rate=18, depth_activation=True)
        image_pooling = AveragePooling2D(8)
        conv_pool = Conv2D(256, (1, 1), padding='same', name='conv_imgPool')
        bn_pool = BatchNormalization(name='imgPool_BN')
        concat1 = Concatenate()
        encoder_op = Conv2D(256, (1, 1), padding='same', name='conv_encoder_op')
        bn_enc = BatchNormalization(name='encoder_op_BN')

        upsample1 = UpSampling2D(size=4)
        conv_low = Conv2D(48, (1, 1), padding='same', name='conv_lowlevel_f')
        bn_low = BatchNormalization(name='low_BN')
        concat2 = Concatenate()
        sepconv_last = self.SeparableConv_BN(filters=256, prefix='final_sepconv', stride=1, depth_activation=True)

        out_conv = Conv2D(self.n_classes, (1, 1), activation='sigmoid', padding='same', name='output_layer')
        upsample2 = UpSampling2D(size=4)

    def call(self, inputs):
        # ===================#
        #  Encoder Network  #
        # ===================#
        # Entry Block
        x = self.conv2d1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.custom_conv1(x)
        x = self.bn2(x)

        x = self.entry_xception1(x)
        x, skip1 = self.entry_xception2(x)
        x = self.entry_xception3(x)

        # Middle Block
        for i in range(16):
            x = self.middle_xception[i](x)

        # Exit Block
        x = self.exit_xception1(x)
        x = self.exit_xception2(x)

        # ====================#
        # Feature Projection #
        # ====================#

        b0 = self.conv_feat(x)
        b0 = self.bn_feat(b0)
        b0 = tf.nn.relu(b0)

        b1 = self.atrous_conv1(x)
        b2 = self.atrous_conv2(x)
        b3 = self.atrous_conv3(x)

        # Image Pooling
        b4 = self.image_pooling(x)
        b4 = self.conv_pool(b4)
        b4 = self.bn_pool(b4)
        b4 = tf.nn.relu(b4)
        b4 = tf.image.resize(b4, size=[b3.get_shape()[1], b3.get_shape()[2]])

        x = self.concat1([b4, b0, b1, b2, b3])

        x = self.encoder_op(x)
        x = self.bn_enc(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, rate=0.1)

        # ===================#
        #  Decoder Network  #
        # ===================#

        x = self.upsample1(x)

        low_level = self.conv_low(skip1)
        low_level = self.bn_low(low_level)
        low_level = tf.nn.relu(low_level)
        x = self.concat2([x, low_level])

        x = self.sepconv_last(x)

        x = self.out_conv(x)
        x = self.upsample2(x)
        return x