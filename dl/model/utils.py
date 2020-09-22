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
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
        """
        data_gen_args = dict(rescale=1. / 255, rotation_range=90, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.05, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        image_generator = image_datagen.flow_from_directory(
            settings.DL_DATASET,
            class_mode=None,
            classes=['image'],
            save_prefix='image',
            save_to_dir=None,
            target_size=(settings.DL_PARAM[network_type]['input_size_w'],
                         settings.DL_PARAM[network_type]['input_size_h']),
            seed=settings.DL_PARAM[network_type]['seed'],
            color_mode=settings.DL_PARAM[network_type]['color_mode'],
            batch_size=settings.DL_PARAM[network_type]['batch_size'],
            shuffle=True)
        mask_generator = mask_datagen.flow_from_directory(
            settings.DL_DATASET,
            class_mode=None,
            classes=['label'],
            save_prefix='ann',
            save_to_dir=None,
            target_size=(settings.DL_PARAM[network_type]['input_size_w'],
                         settings.DL_PARAM[network_type]['input_size_h']),
            seed=settings.DL_PARAM[network_type]['seed'],
            color_mode=settings.DL_PARAM[network_type]['color_mode'],
            batch_size=settings.DL_PARAM[network_type]['batch_size'],
            shuffle=True)

        train_generator = zip(image_generator, mask_generator)

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




