import os
import logging
import imageio
import cv2
import numpy as np
import input.loader as loader
import output.slicer as slicer
import output.poligonize as poligonizer
import settings

from tensorflow.keras.preprocessing import image


class Infer:
    def __init__(self):
        pass

    def save_prediction(self, file_path, output_path, prediction, classes):
        """
        """
        img_color = np.zeros((256, 256, 3), dtype='uint8')
        for i in range(256):
            for j in range(256):
                img_color[i, j] = classes[prediction[i, j]]

        filename = os.path.basename(file_path)
        name, file_extension = os.path.splitext(filename)
        prediction_path = os.path.join(output_path, name + '.png')
        imageio.imwrite(prediction_path, img_color)

        return prediction_path

    def segment_image(self, predicts, classes):
        """
        According to the keras deep learning model, predict (pixelwise image segmentation) the file_path according
        to the classes presented in classes. The outputs are then placed in output_path

        :param predicts:
        :param classes: the list of classes and respectively colors
        """
        predictions_painted_list = []

        for pred in predicts:
            output = np.argmax(pred, axis=-1)
            output = np.expand_dims(output, axis=-1)

            img_color = np.zeros((256, 256, 3), dtype='uint8')
            for i in range(256):
                for j in range(256):
                    img_color[i, j] = classes[output[i, j][0]]
            predictions_painted_list.append(img_color)

        return predictions_painted_list

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

    def display_mask(self, file, prediction, classes, output_path):
        """Quick utility to display a model's prediction."""
        mask = np.argmax(prediction, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        self.save_prediction(file, output_path, mask, classes)

    def make_prediction_and_save(self, model, load_param):
        logging.info(">> Performing prediction...")

        path_val_images = os.path.join(load_param['image_validation_folder'], 'image')
        path_val_labels = os.path.join(load_param['image_validation_folder'], 'label')

        logging.info(">> Loading validation's image datasets...")
        val_images = loader.Loader(path_val_images)

        logging.info(">> Loading validation's label datasets...")
        val_labels = loader.Loader(path_val_labels)
        # val_generator_obj = helper.Helper(16, (256, 256), val_images.get_list_images(),
        #                                   val_labels.get_list_images())

        # val_preds = model.get_model().predict(val_generator_obj)

        list_of_prediction_files = val_images.get_list_images()
        for item in list_of_prediction_files:
            x = cv2.imread(item, cv2.IMREAD_COLOR)
            x = cv2.resize(x, (256, 256))
            x = x / 255.0
            x = x.astype(np.float32)

            ## Prediction
            img = image.load_img(item, target_size=(256, 256))
            p = model.get_model().predict(np.expand_dims(x, axis=0))[0]
            p = np.argmax(p, axis=-1)
            p = np.expand_dims(p, axis=-1)
            # p = p * (255 / 3)
            # p = p.astype(np.int32)
            # p = np.concatenate([p, p, p], axis=2)

            # x = x * 255.0
            # x = x.astype(np.int32)

            # h, w, _ = x.shape
            # line = np.ones((h, 5, 3)) * 255

            # final_image = np.concatenate([x, line, p], axis=1)

            filename = os.path.basename(item)
            name, file_extension = os.path.splitext(filename)
            prediction_path = os.path.join(load_param['output_prediction'], name + '.png')

            cv2.imwrite(prediction_path, p)

            # self.display_mask(list_of_prediction_files[i],
            #                   item,
            #                   load_param['color_classes'],
            #                   load_param['output_prediction'])

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

        path_val_images = os.path.join(load_param['image_validation_folder'], 'image')
        pred_images = loader.Loader(path_val_images)

        # pred_images = loader.Loader(load_param['image_prediction_folder'])

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
                    for slice_item in list_images:
                        segmented_image = self.segment_image(model, slice_item, load_param['color_classes'],
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

                    # logging.info(">>>> Polygonizing segmented image...")
                    # self.poligonize(segmented_image,
                    #                 load_param['classes'],
                    #                 load_param['image_prediction_folder'],
                    #                 load_param['output_prediction'])
            else:
                logging.info(">>>> Image prediction fail: {}. Check filename format!".format(filename))

            # TODO:
            #  2. merge the prediction if it was sliced
            #  3. poligonize the merged prediction image
