import os
import settings
import logging


class Loader:
    """
    Source: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    def __init__(self, dir):
        self.list_image = self.list_entries(dir)

    def list_entries(self, dir):
        """
        """
        logging.info(">> Loading input datasets...")
        input_img_paths = sorted(
            [
                os.path.join(dir, fname)
                for fname in os.listdir(dir)
                if fname.endswith(settings.VALID_ENTRIES_EXTENSION)
            ]
        )

        logging.info(">>>> Number of samples: {}".format(len(input_img_paths)))
        return input_img_paths

    def get_list_images(self):
        return self.list_image
