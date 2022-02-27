import os
import csv
import glob
import logging
import settings
import pandas as pd

from file import image
from file import video
from file import image_utils as iu
from file import video_utils as vu
from lxml import etree


class Reading:
    """ A set of routines to handle IO video and image file """

    def __init__(self):
        pass

    def is_video_filename_in_cvat_xml(self, video_filename, xml_root):
        """
        :param video_filename:
        :param xml_root:
        :return:
        """
        path, file = os.path.split(video_filename)

        for video in xml_root[1].iter('source'):
            if video.text != file:
                return False
            else:
                return True

    def is_a_xml_video_type(self, xml_root):
        """ Check that there is video file in <source> """
        for video in xml_root[1].iter('source'):
            return video.text
        return ''

    def automl2local(self, filename, image_path):
        """
        :param filename:
        :param image_path:
        :return:
        """
        filename = filename.replace('\\', os.sep)
        filename = os.path.basename(filename)

        local_path = os.path.join(image_path, filename)
        return local_path

    def change_filepath(self, csv_input, csv_output, image_path):
        """
        :param csv_input:
        :param csv_output:
        :param image_path:
        :return:
        """
        csv_in = csv.reader(open(csv_input))
        csv_out = csv.writer(open(csv_output, 'w'))
        lines = list(csv_in)

        for i, line in enumerate(lines):
            lines[i][1] = self.automl2local(line[1], image_path)
        csv_out.writerows(lines)

    def choose_entry_files(self, entry_path):
        """
        :param entry_path:
        :return entry_list, entry_type:
        """
        csv_entry_list = self.get_entry_list_by_extension(entry_path, ['.csv', '.CSV'])
        xml_entry_list = self.get_entry_list_by_extension(entry_path, ['.xml', '.XML'])

        entry_type = ''
        entry_list = []
        if len(xml_entry_list) != 0:
            logging.info(">>>> {} XML files available. CSV discarted!".format(len(xml_entry_list)))
            entry_type = 'xml'
            entry_list = xml_entry_list
        elif len(csv_entry_list) != 0 and len(xml_entry_list) == 0:
            logging.info(">>>> {} CSV files available. No XML file found!".format(len(csv_entry_list)))
            entry_type = 'csv'
            entry_list = csv_entry_list
        else:
            logging.error(">>>> There are no CSV either XML files in {}. Check it and try again".format(entry_path))
            exit(0)
        return entry_list, entry_type

    def get_entry_list_by_extension(self, entry_path, accepted_extentions):
        """
        Search all CSVs files in a directory and return the absolute paths in a list

        :param entry_path: A directory with CSVs or XML on it
        :param accepted_extentions: a list of acceptable extensions
        :return entry_list: A list of all acceptable files found in entry_path
        """
        entry_list = []
        logging.info(">> Searching for {} extension files in {}...".format(accepted_extentions, entry_path))

        for ext in accepted_extentions:
            entry_list += glob.glob(entry_path + "/**/*" + ext, recursive=True)

        logging.info(">> Done! {} file(s) found".format(len(entry_list)))
        return entry_list

    def read_cvat_xml_data(self, entry_path, filepath, project_id, tile_image, tile_size):
        """
        :param entry_path:
        :param filepath:
        :param project_id:
        :param tile_image:
        :param tile_size:
        :return:
        """
        logging.info(">> Processing XML file: {}".format(filepath))

        tree = etree.parse(filepath)
        xml_root = tree.getroot()

        is_video = self.is_a_xml_video_type(xml_root)

        if is_video != '':
            file_obj = video.Video(entry_path, filepath, xml_root, tile_image, tile_size)
            vu.VideoUtils(file_obj)
        else:
            file_obj = image.Image(entry_path, filepath, xml_root, tile_image, tile_size)
            iu.ImageUtils(file_obj)

        if len(file_obj.rows) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(file_obj.rows)
        df.columns = settings.COLUMNS_NAME
        df['project'] = project_id

        return df
