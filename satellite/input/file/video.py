import os
import logging

from cv2 import cv2


class Video:
    """ Utils methods to deal with video cvat xml preprocessing """

    def __init__(self, entry_path, xml_filename, xml_root, tile_image, tile_size):
        self.rows = []
        self.images = []

        self.entry_path = entry_path
        self.xml_dirname = os.path.dirname(xml_filename)
        self.xml_filename = xml_filename
        self.xml_root = xml_root
        self.tile_image = tile_image
        self.tile_size = int(tile_size)

        video_source = self.xml_root[1].find('source')
        if video_source is None:
            self.video_filename = ''
        else:
            self.video_filename = os.path.join(self.xml_dirname, video_source.text)

        self.vidcap = cv2.VideoCapture(self.video_filename)
        if self.vidcap.isOpened() is False:
            logging.error(">>>>>> Failed to open video")
            exit(0)

        self.frames = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_labels = [[] for _ in range(int(self.frames))]
        self.used_tiles = {x: set() for x in range(int(self.frames))}
