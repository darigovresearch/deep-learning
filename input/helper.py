import numpy as np
import settings
import imgaug as ia

from imgaug import augmenters as iaa
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img


class Helper(Sequence):
    """
    Helper to iterate over the data (as Numpy arrays)

    Sources:
        - https://keras.io/examples/vision/oxford_pets_image_segmentation/
        - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        - https://stackoverflow.com/questions/43884463/how-to-convert-rgb-image-to-one-hot-encoded-3d-array-based-on-color-using-numpy
        - https://stackoverflow.com/questions/54011487/typeerror-unsupported-operand-types-for-image-and-int
    """

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, shuffle=False, augment=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.input_img_paths))
        # self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates indexes after each epoch

        Source:
            - https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """
        :return: the right proportion of batch trainings, according to the total of images
        """
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        :param idx: batch index
        :return: tuple (input, target) correspond to batch idx.
        """
        indexes = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[indexes: indexes + self.batch_size]
        batch_target_img_paths = self.target_img_paths[indexes: indexes + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = load_img(path, target_size=self.img_size)

        if self.augment is True:
            x = self.augmentor(x)

        if settings.LABEL_TYPE == 'rgb':
            y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
            for j, path in enumerate(batch_target_img_paths):
                img = load_img(path, target_size=self.img_size)
                y[j] = img
        else:
            y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
            for j, path in enumerate(batch_target_img_paths):
                img = load_img(path, target_size=self.img_size, color_mode="grayscale")
                y[j] = np.expand_dims(img, 2)
        return x, y

    def augmentor(self, images):
        """
        Apply data augmentation

        Source:
            - https://github.com/aleju/imgaug
            - https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
        """
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -45 to +45 degrees
                    shear=(-5, 5),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
                                                      n_segments=(20, 200))),
                            # convert images into their superpixel representation
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 1.0)),
                                # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(3, 5)),
                                # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 5)),
                                # blur image using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                            # sharpen images
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                            # emboss images
                            # search either for all edges or for directed edges,
                            # blend the result with the original image using a blobby mask
                            iaa.SimplexNoiseAlpha(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                                       direction=(0.0, 1.0)),
                            ])),
                            iaa.AdditiveGaussianNoise(loc=0,
                                                      scale=(0.0, 0.01 * 255),
                                                      per_channel=0.5),
                            # add gaussian noise to images
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.01, 0.03),
                                                  size_percent=(0.01, 0.02),
                                                  per_channel=0.2),
                            ]),
                            iaa.Invert(0.01, per_channel=True),
                            # invert color channels
                            iaa.Add((-2, 2), per_channel=0.5),
                            # change brightness of images (by -10 to 10 of original value)
                            iaa.AddToHueAndSaturation((-1, 1)),
                            # change hue and saturation
                            # either change the brightness of the whole image (sometimes
                            # per channel) or change the brightness of subareas
                            iaa.OneOf([
                                iaa.Multiply((0.9, 1.1), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-1, 0),
                                    first=iaa.Multiply((0.9, 1.1),
                                                       per_channel=True),
                                    second=iaa.ContrastNormalization(
                                        (0.9, 1.1))
                                )
                            ]),
                            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                                                                sigma=0.25)),
                            # move pixels locally around (with random strengths)
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                            # sometimes move parts of the image around
                            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                            ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        return seq.augment_images(images)
