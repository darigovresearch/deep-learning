import os
import numpy as np
import skimage.io as io
import logging

import settings

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


class DL:
    def __init__(self):
        pass

    # TODO: refactore
    def merge_images(self, paths, max_width, max_height):
        new_im = Image.new('RGB', (max_width, max_height))
        x = 0
        y = 0

        for file in paths:
            img = Image.open(file)
            width, height = img.size
            img.thumbnail((width, height), Image.ANTIALIAS)
            new_im.paste(img, (x, y, x + width, y + height))

            if (x + width) >= max_width:
                x = 0
                y += height
            else:
                x += width

        return new_im

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

    def training_generator(self, network_type, is_augment):
        """
        can generate image and mask at the same time
        hierarchy of folders: https://stackoverflow.com/questions/58050113/imagedatagenerator-for-semantic-segmentation
        :param network_type:
        :param is_augment:
        :return train_generator:
        """
        if is_augment is True:
            data_gen_args = dict(rescale=1./255, rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.15,
                                 fill_mode='nearest')
        else:
            data_gen_args = dict(rescale=1. / 255, validation_split=0.15)

        train_datagen = ImageDataGenerator(**data_gen_args)

        train_image_generator = train_datagen.flow_from_directory(
            settings.DL_PARAM[network_type]['image_training_folder'],
            classes=['image'],
            class_mode=settings.DL_PARAM[network_type]['class_mode'],
            target_size=(settings.DL_PARAM[network_type]['input_size_w'],
                         settings.DL_PARAM[network_type]['input_size_h']),
            seed=settings.DL_PARAM[network_type]['seed'],
            color_mode=settings.DL_PARAM[network_type]['color_mode'],
            batch_size=settings.DL_PARAM[network_type]['batch_size'],
            shuffle=True)
        train_label_generator = train_datagen.flow_from_directory(
            settings.DL_PARAM[network_type]['annotation_training_folder'],
            classes=['label'],
            class_mode=settings.DL_PARAM[network_type]['class_mode'],
            target_size=(settings.DL_PARAM[network_type]['input_size_w'],
                         settings.DL_PARAM[network_type]['input_size_h']),
            seed=settings.DL_PARAM[network_type]['seed'],
            color_mode='grayscale',
            batch_size=settings.DL_PARAM[network_type]['batch_size'],
            shuffle=True)

        train_generator = zip(train_image_generator, train_label_generator)

        return train_generator

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




