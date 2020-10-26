import os
import numpy as np
import logging
import skimage.io as io

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
            io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)

    def label(self, color_dict, img):
        """
        """
        img = img[:, :, 0] if len(img.shape) == 3 else img
        img_out = np.zeros(img.shape + (3,))
        for i in range(len(color_dict)):
            img_out[img == i, :] = color_dict[i]
        return img_out / 255

    # TODO: refactore
    def slice(self, file, width, height, outputFolder):
        valid_images = [".jpg", ".gif", ".png", ".tga", ".tif", ".tiff", ".geotiff"]
        filename = os.path.basename(file)
        name, file_extension = os.path.splitext(filename)

        if file_extension.lower() not in valid_images:
            logging.info(">> Image formats accept: " + str(valid_images) + ". Check image and try again!")

        ds = gdal.Open(file)
        if ds is None:
            raise RuntimeError

        rows = ds.RasterXSize
        cols = ds.RasterYSize

        cont = 0
        paths = []
        gdal.UseExceptions()
        for j in range(0, cols, height):
            for i in range(0, rows, width):
                try:
                    paths.append(outputFolder + name + "_" + "{:05d}".format(cont) + file_extension)
                    com_string = "gdal_translate -eco -q -of GTiff -ot UInt16 -srcwin " + str(i) + " " + str(
                        j) + " " + str(width) + " " + str(
                        height) + " " + file + " " + outputFolder + name + "_" + "{:05d}".format(cont) + file_extension
                    os.system(com_string)
                    cont += 1
                except:
                    pass

        return paths

    def predict_deep_network(self, model, path_in, path_out, path_chp):
        """
        """
        logging.info(">> Performing prediction...")
        file_list = []

        for path in os.listdir(path_in):
            full_path = os.path.join(path_in, path)

            if os.path.isfile(full_path):
                file_list.append(full_path)

        model.predict_multiple(checkpoints_path=path_chp, inps=file_list,  out_dir=path_out)
