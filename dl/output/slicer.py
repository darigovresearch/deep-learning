import logging
import os
import gdal

from os.path import basename
from PIL import Image


class Slicer:
    """
    """
    def __init__(self):
        pass

    def slice(self, file, width, height, output_folder):
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
        paths = []
        rows = ds.RasterXSize
        cols = ds.RasterYSize
        gdal.UseExceptions()
        for j in range(0, cols, height):
            for i in range(0, rows, width):
                try:
                    output_file = os.path.join(output_folder, name + "_" + "{:05d}".format(cont) + file_extension)
                    gdal.Translate(output_file, ds, format='GTIFF', srcWin=[i, j, width, height],
                                   outputType=gdal.GDT_UInt16, options=['TILED=YES'])

                    paths.append(output_file)
                    cont += 1
                except RuntimeError:
                    logging.info(">>>>>> Something went wrong during image slicing...")
        return paths

    def flush_tmps(self, paths):
        """
        Remove all files in paths from the filesystem

        :param paths: list of absolute paths to be removed from filesystem
        """
        for item in paths:
            if os.path.isfile(item):
                os.remove(item)

    def merge_images(self, paths, max_width, max_height):
        """
        Merge the result of each tile in a single image [reverse operation of slice method]

        :param paths: list of absolute paths
        :param max_width: the desired tile width
        :param max_height: the desired tile height
        :return new_im: the merged image
        """
        x = 0
        y = 0
        new_im = Image.new('RGB', (max_width, max_height))

        for index, file in enumerate(paths):
            img = Image.open(file)
            width, height = img.size
            img.thumbnail((width, height), Image.ANTIALIAS)
            new_im.paste(img, (x, y, x + width, y + height))

            if (x+width) >= max_width:
                x = 0
                y += height
            else:
                x += width
        return new_im
