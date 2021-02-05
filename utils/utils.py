import os
import logging


class Utils:
    """
    Utils method for IO and image processing processing
    """
    def __init__(self):
        """
        Constructor method
        """
        pass

    def flush_list_of_paths(self, paths):
        """
        Remove all files from the paths in list

        :param paths: list of absolute paths to be removed from filesystem
        """
        for item in paths:
            if os.path.isfile(item):
                os.remove(item)

    def flush_files(self, folder_path):
        """
        Remove all files inside paths

        :param folder_path: absolute path of a folder
        """
        if not os.path.isdir(folder_path):
            logging.info(">>>> {} is not a directory.".format(folder_path))

        for item in os.listdir(folder_path):
            complete_path = os.path.join(folder_path, item)
            if os.path.isfile(complete_path):
                os.remove(complete_path)
