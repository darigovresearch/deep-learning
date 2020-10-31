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
        :param file:
        :param width:
        :param height:
        :param output_folder:
        :return
        """
        logging.info(">>>> Slicing image " + file + "...")

        filename = basename(file)
        name, file_extension = os.path.splitext(filename)

        if not os.path.isfile(file):
            logging.info(">>>>>> Image {} does not exist. Check it and try again!".format(filename))
            return

        ds = gdal.Open(file)
        if ds is None:
            logging.info(">>>>>> Could not open image file!")
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
        :param paths:
        """
        for item in paths:
            if os.path.isfile(item):
                os.remove(item)

    def merge_images(self, paths, max_width, max_height):
        """
        :param paths:
        :param max_width:
        :param max_height:
        :return
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
