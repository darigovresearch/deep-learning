import os
import numpy as np
import skimage.io as io

import settings

from keras.preprocessing.image import ImageDataGenerator


class DL:
    def __init__(self):
        pass

    def training_generator(self, network_type):
        """
        can generate image and mask at the same time
        hierarchy of folders: https://stackoverflow.com/questions/58050113/imagedatagenerator-for-semantic-segmentation
        """
        data_gen_args = dict(rescale=1./255, rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2,
                             fill_mode='nearest')

        train_datagen = ImageDataGenerator(**data_gen_args)
        val_datagen = ImageDataGenerator(**data_gen_args)

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
            color_mode=settings.DL_PARAM[network_type]['color_mode'],
            batch_size=settings.DL_PARAM[network_type]['batch_size'],
            shuffle=True)

        val_image_generator = val_datagen.flow_from_directory(
            settings.DL_PARAM[network_type]['image_validation_folder'],
            classes=['image'],
            class_mode=settings.DL_PARAM[network_type]['class_mode'],
            target_size=(settings.DL_PARAM[network_type]['input_size_w'],
                         settings.DL_PARAM[network_type]['input_size_h']),
            seed=settings.DL_PARAM[network_type]['seed'],
            color_mode=settings.DL_PARAM[network_type]['color_mode'],
            batch_size=settings.DL_PARAM[network_type]['batch_size'],
            shuffle=True)
        val_label_generator = val_datagen.flow_from_directory(
            settings.DL_PARAM[network_type]['annotation_validation_folder'],
            classes=['label'],
            class_mode=settings.DL_PARAM[network_type]['class_mode'],
            target_size=(settings.DL_PARAM[network_type]['input_size_w'],
                         settings.DL_PARAM[network_type]['input_size_h']),
            seed=settings.DL_PARAM[network_type]['seed'],
            color_mode=settings.DL_PARAM[network_type]['color_mode'],
            batch_size=settings.DL_PARAM[network_type]['batch_size'],
            shuffle=True)

        train_generator = zip(train_image_generator, train_label_generator)
        val_generator = zip(val_image_generator, val_label_generator)

        return train_generator, val_generator

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




