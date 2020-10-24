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

    def adjust_data(self, img, mask, flag_multi_class, num_class):
        if flag_multi_class:
            img = img / 255
            mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
            new_mask = np.zeros(mask.shape + (num_class,))
            for i in range(num_class):
                new_mask[mask == i, i] = 1
            new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                             new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
            mask = new_mask
        elif np.max(img) > 1:
            img = img / 255
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        return img, mask

    def training_generator(self, network_type, is_augment):
        """
        can generate image and mask at the same time
        Source: https://github.com/zhixuhao/unet/blob/master/data.py
        hierarchy of folders: https://stackoverflow.com/questions/58050113/imagedatagenerator-for-semantic-segmentation
        color_mode: https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class
        :param network_type:
        :param is_augment:
        :return train_generator:
        """
        if is_augment is True:
            data_gen_args = dict(rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
        # else:
        #     data_gen_args = dict(rescale=1. / 255)

        load_param = settings.DL_PARAM[network_type]
        
        train_datagen = ImageDataGenerator(**data_gen_args)
        val_datagen = ImageDataGenerator(**data_gen_args)

        train_image_generator = train_datagen.flow_from_directory(
            load_param['image_training_folder'],
            classes=['image'],
            class_mode=load_param['class_mode'],
            target_size=(load_param['input_size_w'],
                         load_param['input_size_h']),
            seed=load_param['seed'],
            color_mode=load_param['color_mode'],
            batch_size=load_param['batch_size'],
            shuffle=True)
        train_label_generator = train_datagen.flow_from_directory(
            load_param['annotation_training_folder'],
            classes=['label'],
            class_mode=load_param['class_mode'],
            target_size=(load_param['input_size_w'],
                         load_param['input_size_h']),
            seed=load_param['seed'],
            color_mode='grayscale',
            batch_size=load_param['batch_size'],
            shuffle=True)

        val_image_generator = val_datagen.flow_from_directory(
            load_param['image_validation_folder'],
            classes=['image'],
            class_mode=load_param['class_mode'],
            target_size=(load_param['input_size_w'],
                         load_param['input_size_h']),
            seed=load_param['seed'],
            color_mode=load_param['color_mode'],
            batch_size=load_param['batch_size'],
            shuffle=True)
        val_label_generator = val_datagen.flow_from_directory(
            load_param['annotation_validation_folder'],
            classes=['label'],
            class_mode=load_param['class_mode'],
            target_size=(load_param['input_size_w'],
                         load_param['input_size_h']),
            seed=load_param['seed'],
            color_mode='grayscale',
            batch_size=load_param['batch_size'],
            shuffle=True)

        train_generator = zip(train_image_generator, train_label_generator)
        val_generator = zip(val_image_generator, val_label_generator)

        # for img, mask in train_generator:
        #     img, mask = self.adjust_data(img, mask, True, len(load_param['classes']))
        #     yield img, mask
        #
        # for img, mask in val_generator:
        #     img, mask = self.adjust_data(img, mask, True, len(load_param['classes']))
        #     yield img, mask

        return train_generator, val_generator
