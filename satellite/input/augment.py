import os
import logging
import numpy as np
from satellite import settings

from PIL import Image
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import load_img


class Augment:

    def __init__(self, img_size, train_image_paths, train_labels_paths):
        self.img_size = img_size
        self.train_image_paths = train_image_paths
        self.train_labels_paths = train_labels_paths

    def get_augment_seq(self, aug_type):
        """
        Prepare augmentation and return the respective aug_type object

        :param aug_type: string describing the augmentation type
        :return: imgaug object

        Source:
            - https://www.programcreek.com/python/example/115046/imgaug.HooksImages
            - https://www.programcreek.com/python/?code=JohnleeHIT%2FBrats2019%2FBrats2019-master%2Fsrc%2Futils.py#
        """
        if aug_type == 'all':
            seq = iaa.Sequential([
                iaa.Fliplr(1.0),  # horizontally flip 50% of all images
                iaa.Flipud(1.0),  # vertically flip 20% of all images
                iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
                iaa.Affine(scale=(0.5, 0.8)),
                iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
                iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            ], random_order=True)
        elif aug_type == 'rotation':
            seq = iaa.Sequential([
                iaa.Fliplr(1.0),  # horizontally flip 50% of all images
                iaa.Flipud(1.0),  # vertically flip 20% of all images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            ], random_order=True)
        elif aug_type == 'noise':
            seq = iaa.Sequential([
                iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
            ], random_order=True)
        elif aug_type == 'blured':
            seq = iaa.Sequential([
                iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
            ], random_order=True)
        elif aug_type == 'resize':
            seq = iaa.Sequential([
                iaa.Affine(scale=(0.5, 0.8))
            ], random_order=True)
        else:
            seq = iaa.Sequential([
                iaa.Affine(scale=0.5),  # rotate by -45 to 45 degrees (affects segmaps)
            ], random_order=True)

        return seq

    def image_aug_filename(self, path, aug_type):
        """
        Setup filename to the augmented image

        :param path: absolute path to original file
        :param aug_type: type of augment effect
        :return: the new augmented image filename
        """
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        name, extension = filename.split('.')
        image_aug_filename = os.path.join(dirname, name + "_aug_" + aug_type + "." + extension)

        return image_aug_filename

    def augment(self):
        """
        Get all images entries and apply augmentation according to types variable
        """
        types = ['all', 'rotation', 'blured', 'noise', 'resize']

        for t in types:
            seq = self.get_augment_seq(t)
            det = seq.to_deterministic()

            logging.info(">>>> Augmenting with {} effects...".format(t))

            for j in range(0, len(self.train_image_paths)):
                x = np.zeros(self.img_size + (3,), dtype="float32")
                x = load_img(self.train_image_paths[j], target_size=self.img_size)
                x = np.asarray(x)

                if settings.LABEL_TYPE == 'rgb':
                    y = np.zeros(self.img_size + (3,), dtype="uint8")
                    y = load_img(self.train_labels_paths[j], target_size=self.img_size)
                else:
                    y = load_img(self.train_labels_paths[j], target_size=self.img_size, color_mode="grayscale")
                    y = np.expand_dims(y, 2)

                x = det.augment_image(x)
                y = det.augment_image(y)

                image_aug_filename = self.image_aug_filename(self.train_image_paths[j], t)
                label_aug_filename = self.image_aug_filename(self.train_labels_paths[j], t)

                im_x = Image.fromarray(x)
                y = np.squeeze(y, axis=2)
                im_y = Image.fromarray(y)

                im_x.save(image_aug_filename, "TIFF")
                im_y.save(label_aug_filename)
