import os
import logging
import imageio
import cv2
import numpy as np

import dl.input.loader as loader
import dl.output.slicer as slicer
import dl.output.poligonize as poligonizer
import settings


class Infer:
    def __init__(self):
        pass

    def segment_image(self, model, file_path, classes, output_path):
        """
        According to the keras deep learning model, predict (pixelwise image segmentation) the file_path according
        to the classes presented in classes. The outputs are then placed in output_path

        :param model: the compiled keras deep learning architecture
        :param file_path: absolute path to the image file
        :param classes: the list of classes and respectively colors
        :param output_path: the absolute path for predictions
        """
        image_full = cv2.imread(file_path)
        dims = image_full.shape
        image_full = image_full / 255
        image_full = np.reshape(image_full, (1, dims[0], dims[1], dims[2]))

        pr = model.get_model().predict(image_full)
        output = np.argmax(pr, axis=-1)
        output = np.reshape(output, (dims[0], dims[1]))

        img_color = np.zeros((dims[0], dims[1], dims[2]), dtype='uint8')
        for i in range(dims[0]):
            for j in range(dims[1]):
                img_color[i, j] = classes[output[i, j]]

        filename = os.path.basename(file_path)
        name, file_extension = os.path.splitext(filename)
        prediction_path = os.path.join(output_path, name + '.png')
        imageio.imwrite(prediction_path, img_color)

        return prediction_path

    def poligonize(self, segmented, classes, original_images_path, output_vector_path):
        """
        Turn a JPG, PNG images in a geographic format, such as ESRI Shapefile or GeoJSON. The image must to be
        in the exact colors specified in settings.py [DL_PARAM['classes']]

        :param segmented: the segmented image path
        :param classes: the list of classes and respectively colors
        :param original_images_path: the path to the original images, certainly, with the geographic metadata
        :param output_vector_path: the output path file to save the respective geographic format
        """
        if isinstance(segmented, list):
            for item in segmented:
                poligonizer.Poligonize().polygonize(item, classes, original_images_path, output_vector_path)
        else:
            poligonizer.Poligonize().polygonize(segmented, classes, original_images_path, output_vector_path)

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

        pred_images = loader.Loader(load_param['image_prediction_folder'])

        for item in pred_images.get_list_images():
            filename = os.path.basename(item)

            if filename.endswith(settings.VALID_PREDICTION_EXTENSION):
                image_full = cv2.imread(item)
                dims = image_full.shape

                if dims[0] > load_param['width_slice'] or dims[1] > load_param['height_slice']:
                    logging.info(">>>> Image {} is bigger than the required dimension! Cropping and predicting...")
                    list_images = slicer.Slicer().slice(image_full, load_param['width_slice'],
                                                        load_param['height_slice'],
                                                        load_param['image_prediction_tmp_slice_folder'])

                    segmented_slices_list = []
                    for item in list_images:
                        segmented_image = self.segment_image(model, item, load_param['color_classes'],
                                                             load_param['output_prediction'])
                        segmented_slices_list.append(segmented_image)

                    logging.info(">>>> Sewing segmented image and polygonizing...")
                    self.poligonize(segmented_slices_list,
                                    load_param['classes'],
                                    load_param['image_prediction_folder'],
                                    load_param['output_prediction'])

                else:
                    segmented_image = self.segment_image(model, item, load_param['color_classes'],
                                                         load_param['output_prediction'])

                    logging.info(">>>> Polygonizing segmented image...")
                    self.poligonize(segmented_image,
                                    load_param['classes'],
                                    load_param['image_prediction_folder'],
                                    load_param['output_prediction'])
            else:
                logging.info(">>>> Image prediction fail: {}. Check filename format!".format(filename))

            # TODO:
            #  2. merge the prediction if it was sliced
            #  3. poligonize the merged prediction image
