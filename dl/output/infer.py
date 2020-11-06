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

    # def poligonize(self, output):
    #     """
    #     """
    #     if isinstance(output, list):
    #         for item in output:
    #             poligonizer.Poligonize().polygonize(item, image, output)
    #     else:
    #         poligonizer.Poligonize().polygonize(output, image, output)

    def predict_deep_network(self, model, load_param):
        """
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

                    output_slices_list = []
                    for item in list_images:
                        output = self.segment_image(model, item, load_param['color_classes'],
                                                    load_param['output_prediction'])
                        output_slices_list.append(output)

                    logging.info(">>>> Sewing results and poligonizing...")
                    # self.poligonize(output_slices_list)

                else:
                    output = self.segment_image(model, item, load_param['color_classes'],
                                                load_param['output_prediction'])
                    logging.info(">>>> Sewing results and poligonizing...")
                    # self.poligonize(output)
            else:
                logging.info(">>>> Image prediction fail: {}. Check filename format!".format(filename))

            # TODO:
            #  2. merge the prediction if it was sliced
            #  3. poligonize the merged prediction image
