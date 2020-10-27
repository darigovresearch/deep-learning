import os
import logging
import imageio
import cv2
import skimage.io as io
import numpy as np

import dl.model.loader as loader
import dl.model.helper as helper
import settings

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')


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

    def slice(self, file, width, height, output_folder):
        """
        """
        filename = os.path.basename(file)
        name, file_extension = os.path.splitext(filename)

        if file_extension.lower() not in settings.VALID_PREDICTION_EXTENSION:
            logging.info(">> Image formats accept: {}. Check image and try again!".
                         format(settings.VALID_PREDICTION_EXTENSION))

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
                    paths.append(output_folder + name + "_" + "{:05d}".format(cont) + file_extension)
                    com_string = "gdal_translate -eco -q -of GTiff -ot UInt16 -srcwin " + str(i) + " " + str(
                        j) + " " + str(width) + " " + str(
                        height) + " " + file + " " + output_folder + name + "_" + "{:05d}".format(cont) + file_extension
                    os.system(com_string)
                    cont += 1
                except:
                    pass

        return paths

    def predict_deep_network(self, model, load_param):
        """
        """
        logging.info(">> Performing prediction...")

        pred_images = loader.Loader(load_param['image_prediction_folder'])

        for item in pred_images.get_list_images():
            filename = os.path.basename(item)
            name, extension = os.path.splitext(filename)

            if filename.endswith(settings.VALID_PREDICTION_EXTENSION):
                image_full = cv2.imread(item)
                dims = image_full.shape
                image_full = image_full / 255
                image_full = np.reshape(image_full, (1, dims[0], dims[1], dims[2]))

                pr = model.get_model().predict(image_full)
                pred_mask = np.argmax(pr, axis=-1)
                output = np.reshape(pred_mask, (dims[0], dims[1]))

                img_color = np.zeros((dims[0], dims[1], dims[2]), dtype='uint8')
                for j in range(dims[0]):
                    for i in range(dims[1]):
                        img_color[j, i] = load_param['color_classes'][output[j, i]]

                prediction_path = os.path.join(load_param['output_prediction'], name + '.png')
                imageio.imwrite(prediction_path, img_color)
            else:
                logging.info(">>>> Image prediction fail: {}. Check filename format!".format(filename))

            # TODO:
            #  2. merge the prediction if it was sliced
            #  3. poligonize the merged prediction image
