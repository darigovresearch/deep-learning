import os
import numpy as np
import logging
import skimage.io

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

    def save_result(self, save_path, load_param, npyfile, flag_multi_class=False):
        """
        """
        color_dict = load_param['classes']

        for i, item in enumerate(npyfile):
            img = self.label(color_dict, item) if flag_multi_class else item[:, :, 0]
            # io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)

    def label(self, color_dict, img):
        """
        """
        img = img[:, :, 0] if len(img.shape) == 3 else img
        img_out = np.zeros(img.shape + (3,))
        for i in range(len(color_dict)):
            img_out[img == i, :] = color_dict[i]
        return img_out / 255

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
