import os
import numpy as np
import logging
import skimage.io
import imagereader

from dl.model import unet as model

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')


TILE_SIZE = 256


class Infer:
    def __init__(self):
        pass

    def predict_deep_network(self, model, path_in, path_out, path_chp):
        """
        """
        logging.info(">> Perform prediction...")
        file_list = []

        for path in os.listdir(path_in):
            full_path = os.path.join(path_in, path)

            if os.path.isfile(full_path):
                file_list.append(full_path)

        model.predict_multiple(checkpoints_path=path_chp, inps=file_list,  out_dir=path_out)

    def _inference_tiling(self, img, unet_model, tile_size):
        # Pad the input image in CPU memory to ensure its dimensions are multiples of the U-Net Size Factor
        pad_x = 0
        pad_y = 0
        if img.shape[0] % model.UNet.SIZE_FACTOR != 0:
            pad_y = (model.UNet.SIZE_FACTOR - img.shape[0] % model.UNet.SIZE_FACTOR)
            print('image height needs to be a multiple of {}, padding with reflect'.format(model.UNet.SIZE_FACTOR))
        if img.shape[1] % model.UNet.SIZE_FACTOR != 0:
            pad_x = (model.UNet.SIZE_FACTOR - img.shape[1] % model.UNet.SIZE_FACTOR)
            print('image width needs to be a multiple of {}, padding with reflect'.format(model.UNet.SIZE_FACTOR))

        if len(img.shape) != 2 and len(img.shape) != 3:
            raise IOError('Invalid number of dimensions for input image. Expecting HW or HWC dimension ordering.')

        if len(img.shape) == 2:
            # add a channel dimension
            img = img.reshape((img.shape[0], img.shape[1], 1))
        if pad_x > 0 or pad_y > 0:
            img = np.pad(img, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
            print('Padded Image Size: {}'.format(img.shape))

        height = img.shape[0]
        width = img.shape[1]
        mask = np.zeros((height, width), dtype=np.int32)

        # radius = model.UNet.RADIUS  # theoretical radius
        radius = unet_model.estimate_radius()
        print('Estimated radius based on ERF : "{}"'.format(radius))
        assert tile_size % model.UNet.SIZE_FACTOR == 0
        assert radius % model.UNet.SIZE_FACTOR == 0
        zone_of_responsibility_size = tile_size - 2 * radius
        assert zone_of_responsibility_size >= radius

        for i in range(0, height, zone_of_responsibility_size):
            for j in range(0, width, zone_of_responsibility_size):

                x_st_z = j
                y_st_z = i
                x_end_z = x_st_z + zone_of_responsibility_size
                y_end_z = y_st_z + zone_of_responsibility_size

                # pad zone of responsibility by radius
                x_st = x_st_z - radius
                y_st = y_st_z - radius
                x_end = x_end_z + radius
                y_end = y_end_z + radius

                radius_pre_x = radius
                if x_st < 0:
                    x_st = 0
                    radius_pre_x = 0

                radius_pre_y = radius
                if y_st < 0:
                    radius_pre_y = 0
                    y_st = 0

                radius_post_x = radius
                if x_end > width:
                    radius_post_x = 0
                    x_end = width
                    x_end_z = width

                radius_post_y = radius
                if y_end > height:
                    radius_post_y = 0
                    y_end = height
                    y_end_z = height

                # crop out the tile
                tile = img[y_st:y_end, x_st:x_end]

                # convert HWC to CHW
                batch_data = tile.transpose((2, 0, 1))
                # convert CHW to NCHW
                batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))

                sm = unet_model.get_keras_model()(batch_data)  # model output defined in unet_model is softmax
                sm = np.squeeze(sm)
                pred = np.squeeze(np.argmax(sm, axis=-1).astype(np.int32))

                # radius_pre_x
                if radius_pre_x > 0:
                    pred = pred[:, radius_pre_x:]
                    sm = sm[:, radius_pre_x:]

                # radius_pre_y
                if radius_pre_y > 0:
                    pred = pred[radius_pre_y:, :]
                    sm = sm[radius_pre_y:, :]

                # radius_post_x
                if radius_post_x > 0:
                    pred = pred[:, :-radius_post_x]
                    sm = sm[:, :-radius_post_x]

                # radius_post_y
                if radius_post_y > 0:
                    pred = pred[:-radius_post_y, :]
                    sm = sm[:-radius_post_y, :]

                mask[y_st_z:y_end_z, x_st_z:x_end_z] = pred

        # undo and CPU side image padding to make the image a multiple of U-Net Size Factor
        if pad_x > 0:
            mask = mask[:, 0:-pad_x]
        if pad_y > 0:
            mask = mask[0:-pad_y, :]
        return mask

    def _inference(self, img, unet_model):
        pad_x = 0
        pad_y = 0

        if img.shape[0] % model.UNet.SIZE_FACTOR != 0:
            pad_y = (model.UNet.SIZE_FACTOR - img.shape[0] % model.UNet.SIZE_FACTOR)
            print('image height needs to be a multiple of {}, padding with reflect'.format(model.UNet.SIZE_FACTOR))
        if img.shape[1] % model.UNet.SIZE_FACTOR != 0:
            pad_x = (model.UNet.SIZE_FACTOR - img.shape[1] % model.UNet.SIZE_FACTOR)
            print('image width needs to be a multiple of {}, padding with reflect'.format(model.UNet.SIZE_FACTOR))

        if len(img.shape) != 2 and len(img.shape) != 3:
            raise IOError('Invalid number of dimensions for input image. Expecting HW or HWC dimension ordering.')

        if len(img.shape) == 2:
            # add a channel dimension
            img = img.reshape((img.shape[0], img.shape[1], 1))
        img = np.pad(img, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')

        # convert HWC to CHW
        batch_data = img.transpose((2, 0, 1))
        # convert CHW to NCHW
        batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))
        batch_data = tf.convert_to_tensor(batch_data)

        softmax = unet_model.get_keras_model()(batch_data) # model output defined in unet_model is softmax
        softmax = np.squeeze(softmax)
        pred = np.squeeze(np.argmax(softmax, axis=-1).astype(np.int32))

        if pad_x > 0:
            pred = pred[:, 0:-pad_x]
        if pad_y > 0:
            pred = pred[0:-pad_y, :]

        return pred

    def inference(self, unet_model, image_folder, output_folder, image_format):

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

        # unet_model = model.UNet(input_size, number_classes, number_channels, False, False)
        # # unet_model = model.UNet(number_classes, 1, number_channels, 1e-4)
        # unet_model.load_checkpoint(checkpoint_filepath)

        print('Starting inference of file list')
        for i in range(len(img_filepath_list)):
            img_filepath = img_filepath_list[i]
            _, slide_name = os.path.split(img_filepath)
            print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

            print('Loading image: {}'.format(img_filepath))
            img = imagereader.imread(img_filepath)
            img = img.astype(np.float32)

            # normalize with whole image stats
            img = imagereader.zscore_normalize(img)
            print('  img.shape={}'.format(img.shape))

            if img.shape[0] > TILE_SIZE or img.shape[1] > TILE_SIZE:
                segmented_mask = self._inference_tiling(img, unet_model, TILE_SIZE)
            else:
                segmented_mask = self._inference(img, unet_model)

            if 0 <= np.max(segmented_mask) <= 255:
                segmented_mask = segmented_mask.astype(np.uint8)
            if 255 < np.max(segmented_mask) < 65536:
                segmented_mask = segmented_mask.astype(np.uint16)
            if np.max(segmented_mask) > 65536:
                segmented_mask = segmented_mask.astype(np.int32)
            if 'tif' in image_format:
                skimage.io.imsave(os.path.join(output_folder, slide_name), segmented_mask, compress=6, bigtiff=True,
                                  tile=(TILE_SIZE, TILE_SIZE))
            else:
                try:
                    skimage.io.imsave(os.path.join(output_folder, slide_name), segmented_mask, compress=6)
                except TypeError:  # compress option not valid
                    skimage.io.imsave(os.path.join(output_folder, slide_name), segmented_mask)
