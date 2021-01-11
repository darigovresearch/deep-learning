import logging
import os
import gdal
import settings

from os.path import basename
from PIL import Image


class Slicer:
    """
    """
    def __init__(self):
        pass

    def slice_bitmap(self, file, width, height, output_folder):
        """
        Open the non-geographic image file, and crop it equally with dimensions of width x height, placing
        it in output_folder. The remaining borders is also cropped and saved in the folder

        :param file: absolute image path [oversized image]
        :param width: the desired tile width
        :param height: the desired tile height
        :param output_folder: the destination folder of the tiles/slices
        :return paths: a list of absolute paths, regarding each tile cropped
        """
        logging.info(">>>> Slicing image " + file + "...")

        filename = basename(file)
        name, file_extension = os.path.splitext(filename)

        if not os.path.isfile(file):
            logging.info(">>>>>> Image {} does not exist. Check it and try again!".format(filename))
            return

        image = Image.open(file)
        cols, rows = image.size

        cont = 0
        paths = []
        buffer = settings.BUFFER_TO_INFERENCE
        for j in range(0, cols, (height - buffer)):
            for i in range(0, rows, (width - buffer)):
                output_file = os.path.join(output_folder, name + "_" + "{:05d}".format(cont) + file_extension)
                if not ((i + width) > rows) and not ((j + height) > cols):
                    image.crop((i, j, i + width, j + height)).save(output_file)

                    paths.append(output_file)
                    cont += 1
        return paths

    def slice_geographic(self, file, width, height, output_folder):
        """
        Open the image file, and crop it equally with dimensions of width x height, placing it in output_folder.
        The remaining borders is also cropped and saved in the folder

        :param file: absolute image path [oversized image]
        :param width: the desired tile width
        :param height: the desired tile height
        :param output_folder: the destination folder of the tiles/slices
        :return paths: a list of absolute paths, regarding each tile cropped
        """
        logging.info(">>>> Slicing image " + file + "...")

        filename = basename(file)
        name, file_extension = os.path.splitext(filename)

        if not os.path.isfile(file):
            logging.info(">>>>>> Image {} does not exist. Check it and try again!".format(filename))
            return

        ds = gdal.Open(file)
        if ds is None:
            logging.info(">>>>>> Could not open image file. Check it and try again!")
            return

        cont = 0
        rows = ds.RasterXSize
        cols = ds.RasterYSize
        datatype = ds.GetRasterBand(1).DataType

        paths = []
        gdal.UseExceptions()
        buffer = settings.BUFFER_TO_INFERENCE
        for j in range(0, cols, (height - buffer)):
            for i in range(0, rows, (width - buffer)):
                output_file = os.path.join(output_folder, name + "_" + "{:05d}".format(cont) + file_extension)
                try:
                    if not ((i + width) > rows) and not ((j + height) > cols):
                        gdal.Translate(output_file, ds, format='GTIFF', srcWin=[i, j, width, height],
                                       outputType=datatype, options=['-eco', '-epo',
                                                                     '-b', settings.RASTER_TILES_COMPOSITION[0],
                                                                     '-b', settings.RASTER_TILES_COMPOSITION[1],
                                                                     '-b', settings.RASTER_TILES_COMPOSITION[2]])

                        paths.append(output_file)
                        cont += 1
                except RuntimeError:
                    logging.warning(">>>>>> Something went wrong during image slicing...")
        return paths

    def merge_images(self, paths, max_width, max_height, complete_path_to_merged_prediction):
        """
        Merge the result of each tile in a single image [reverse operation of slice method]

        :param paths: list of absolute paths
        :param max_width: the desired tile width
        :param max_height: the desired tile height
        :param complete_path_to_merged_prediction:
        """
        new_im = Image.new('RGB', (max_width, max_height))

        x = 0
        y = 0
        buffer = settings.BUFFER_TO_INFERENCE
        for file in paths:
            img = Image.open(file)
            width, height = img.size

            img.thumbnail((width, height), Image.ANTIALIAS)
            if not ((x + width) > max_width) and not ((y + height) > max_height):
                new_im.paste(img, (x, y, x + width, y + height))
            else:
                x = 0
                y += (height - buffer)
                new_im.paste(img, (x, y, x + width, y + height))
            x += (width - buffer)

        new_im.save(complete_path_to_merged_prediction)
