import os
import settings
import logging


class Loader:
    """
    Image training and validation loader

    Source:
        - https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """

    def __init__(self, dir):
        self.list_image = self.list_entries(dir)

    def list_entries(self, directory):
        """
        Image training and validation loader. List the entries
        
        :param directory: 
        :return input_img_paths: list
        """
        logging.info(">> Loading input datasets...")
        input_img_paths = sorted(
            [
                os.path.join(directory, fname)
                for fname in os.listdir(directory)
                if fname.endswith(settings.VALID_ENTRIES_EXTENSION)
            ]
        )

        logging.info(">>>> Number of samples: {}".format(len(input_img_paths)))
        return input_img_paths

    def get_list_images(self):
        """
        :return list_image: return the list of absolute path to the images
        """
        return self.list_image
