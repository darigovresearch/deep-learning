import os
import settings
import logging


class Loader:
    """
    Source: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    def __init__(self, input_dir, target_dir):
        self.list_image, self.list_label = self.list_entries(input_dir, target_dir)

    def list_entries(self, input_dir, target_dir):
        """
        """
        logging.info(">> Loading input datasets...")
        input_img_paths = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(settings.VALID_ENTRIES_EXTENSION)
            ]
        )

        target_img_paths = sorted(
            [
                os.path.join(target_dir, fname)
                for fname in os.listdir(target_dir)
                if fname.endswith(settings.VALID_ENTRIES_EXTENSION) and not fname.startswith(".")
            ]
        )

        logging.info(">>>> Number of samples:", len(input_img_paths))
        logging.info(">>>>>> 10 first: ")
        for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
            logging.info(">>>>>>>> {} | {}".format(input_path, target_path))

        return input_img_paths, target_img_paths

    def get_list_images(self):
        return self.list_image

    def get_list_labels(self):
        return self.list_label
