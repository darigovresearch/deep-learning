import numpy as np
import settings

from keras.utils import Sequence
from keras.utils import to_categorical
from keras.preprocessing.image import load_img


class Helper(Sequence):
    """
    Helper to iterate over the data (as Numpy arrays)

    Sources:
        - https://keras.io/examples/vision/oxford_pets_image_segmentation/
        - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        - https://stackoverflow.com/questions/43884463/how-to-convert-rgb-image-to-one-hot-encoded-3d-array-based-on-color-using-numpy
        - https://stackoverflow.com/questions/54011487/typeerror-unsupported-operand-types-for-image-and-int
    """

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

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
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = np.asarray(img) / 255

        if settings.LABEL_TYPE == 'classid':
            y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
            for j, path in enumerate(batch_target_img_paths):
                img = load_img(path, target_size=self.img_size, color_mode="grayscale")
                y[j] = np.expand_dims(img, 2)

            # TODO: hardcoded number of classes
            # y = to_categorical(y.astype('float32'), num_classes=3)
        elif settings.LABEL_TYPE == 'rgb':
            y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
            for j, path in enumerate(batch_target_img_paths):
                img = load_img(path, target_size=self.img_size)
                y[j] = img
        return x, y


