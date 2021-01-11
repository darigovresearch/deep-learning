import os
import logging
import imageio
import cv2
import gdal
import shutil
import numpy as np
import input.loader as loader
import output.slicer as slicer
import output.poligonize as poligonizer
import settings

from utils import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


class Infer:
    def __init__(self):
        pass

    def segment_image(self, image_path, prediction, classes, output_path):
        """
        Create a new RGB image, drawing the predictions based on classes's colors
        :param image_path:
        :param prediction:
        :param classes:
        :param output_path:
        :return prediction_path:
        """
        output = np.argmax(prediction, axis=-1)
        output = np.reshape(output, (256, 256))

        img_color = np.zeros((256, 256, 3), dtype='uint8')
        for i in range(256):
            for j in range(256):
                idx = output[i, j]
                img_color[i, j] = classes[idx]

        filename = os.path.basename(image_path)
        name, file_extension = os.path.splitext(filename)
        prediction_path = os.path.join(output_path, name + '.png')
        imageio.imwrite(prediction_path, img_color)
        return prediction_path

    def poligonize(self, segmented_image_path, classes, original_images_path, output_vector_path):
        """
        Turn a JPG, PNG images in a geographic format, such as ESRI Shapefile or GeoJSON. The image must to be
        in the exact colors specified in settings.py [DL_PARAM['classes']]

        :param segmented_image_path: the segmented image path, which could be a list or not
        :param classes: the list of classes and respectively colors
        :param original_images_path: the path to the original images, certainly, with the geographic metadata
        :param output_vector_path: the output path file to save the respective geographic format
        """
        if isinstance(segmented_image_path, list):
            for item in segmented_image_path:
                poligonizer.Poligonize().polygonize(item, classes, original_images_path, output_vector_path)
        else:
            poligonizer.Poligonize().polygonize(segmented_image_path, classes, original_images_path, output_vector_path)

    def check_image_format(self, image_path):
        """
        Based on file extensions, this method determines if it could be treat as a geographic format or not

        :param image_path: absolute path to the original raster image
        :return dims, is_geographic_format: the dimension size of the respective image, and a boolean,
        if it is a geographic format or not
        """
        is_geographic_format = False

        filename = os.path.basename(image_path)
        if filename.endswith(settings.GEOGRAPHIC_ACCEPT_EXTENSION):
            ds = gdal.Open(image_path)
            if ds is None:
                logging.info(">>>>>> Could not open image file. Check it and try again!")
                return
            dims = ds.RasterXSize, ds.RasterYSize
            is_geographic_format = True
        elif filename.endswith(settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION):
            image_full = cv2.imread(image_path)
            dims = image_full.shape
        else:
            return None, None

        return dims, is_geographic_format

    def predict_deep_network(self, model, load_param):
        """
        Initiate the process of inferences. The weight matrix from trained deep learning, which represents the
        knowledge, is loaded and the images are then presented. Each one is processed (multiclass or not) and
        submitted to the polygonization, where the raster is interpreted and a correspondent geographic
        format is created

        :param model: the compiled keras deep learning architecture
        :param load_param: a dict with the keras deep learning architecture parameters
        """
        logging.info(">> Performing prediction...")

        path_val_images = os.path.join(load_param['image_prediction_folder'])
        pred_images = loader.Loader(path_val_images)

        for item in pred_images.get_list_images():
            filename = os.path.basename(item)
            name = os.path.splitext(filename)[0]
            complete_path = os.path.join(path_val_images, item)
            dims, is_geographic_file = self.check_image_format(complete_path)

            if dims is None or is_geographic_file is None:
                logging.warning(">>>>>> The filename {} does not match any accepted extension. "
                                "Check it and try again!".format(filename))
                return

            if dims[0] > load_param['width_slice'] or dims[1] > load_param['height_slice']:
                logging.info(">>>> Image {} is bigger than the required dimension! "
                             "Cropping and predicting...".format(filename))

                if is_geographic_file is True:
                    list_images = slicer.Slicer().slice_geographic(complete_path, load_param['width_slice'],
                                                                   load_param['height_slice'],
                                                                   load_param['tmp_slices'])
                else:
                    list_images = slicer.Slicer().slice_bitmap(complete_path, load_param['width_slice'],
                                                               load_param['height_slice'],
                                                               load_param['tmp_slices'])

                logging.info(">>>> Loading slices and predicting...")

                prediction_path_list = []
                for path in list_images:
                    images_array = load_img(path, target_size=(256, 256))
                    images_array = image.img_to_array(images_array)
                    images_array = np.expand_dims(images_array, axis=0)

                    prediction = model.get_model().predict(images_array)
                    prediction_path_list.append(self.segment_image(path, prediction, load_param['color_classes'],
                                                                   load_param['tmp_slices_predictions']))

                logging.info(">>>> Merging {} predictions in image with {} x {}...".format(len(prediction_path_list),
                                                                                           dims[0], dims[1]))
                complete_path_to_merged_prediction = os.path.join(load_param['output_prediction'], name + ".png")
                slicer.Slicer().merge_images(prediction_path_list, dims[0], dims[1], complete_path_to_merged_prediction)

                # if is_geographic_file is True:
                #     logging.info(">>>> Polygonizing segmented image...")
                #     self.poligonize(complete_path_to_merged_prediction,
                #                     load_param['classes'],
                #                     complete_path,
                #                     load_param['output_prediction_shp'])
                #
                #     utils.Utils().flush_files(load_param['tmp_slices'])
                #     utils.Utils().flush_files(load_param['tmp_slices_predictions'])
            else:
                image_to_predict = load_img(complete_path, target_size=(256, 256))
                images_array = image.img_to_array(image_to_predict)
                images_array = np.expand_dims(images_array, axis=0)

                prediction = model.get_model().predict(images_array)

                prediction_path = self.segment_image(complete_path, prediction, load_param['color_classes'],
                                                     load_param['tmp_slices_predictions'])

                if is_geographic_file is True:
                    logging.info(">>>> Polygonizing segmented image...")
                    self.poligonize(prediction_path,
                                    load_param['classes'],
                                    complete_path,
                                    load_param['output_prediction_shp'])
