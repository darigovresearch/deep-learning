import os


class Image:
    """ Utils methods to deal with image cvat xml preprocessing """

    def __init__(self, entry_path, xml_filename, xml_root, tile_image, tile_size):
        self.rows = []
        self.images = []
        self.used_tiles_list = []

        self.entry_path = entry_path
        self.xml_dirname = os.path.dirname(xml_filename)
        self.xml_filename = xml_filename
        self.xml_root = xml_root
        self.num_images = sum(1 for _ in self.xml_root.iter('image'))
        self.tile_image = tile_image
        self.tile_size = int(tile_size)
