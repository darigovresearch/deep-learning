import cv2
import numpy as np
import settings

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

        Source:
            - https://stackoverflow.com/questions/53235638/how-should-i-convert-a-float32-image-to-an-uint8-image
        """
        indexes = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[indexes: indexes + self.batch_size]
        batch_target_img_paths = self.target_img_paths[indexes: indexes + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = load_img(path, target_size=self.img_size)
            x[j] = cv2.normalize(x[j], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)

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

        # if self.augment is True:
        #     x = x.astype(np.uint8)
        #     x, y = self.augmentor_3(x, y)
        #     x = x.astype(np.float32)

        return x, y
