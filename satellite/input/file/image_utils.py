import os
import logging
import platform

from cv2 import cv2
from . import settings


class ImageUtils:
    """ Utils methods to deal with image cvat xml preprocessing """

    def __init__(self, image_obj):
        self.build_xml_image_object(image_obj)

    def build_xml_image_object(self, image_obj):
        """
        :return:
        """
        num_bboxes = sum(1 for _ in image_obj.xml_root.iter('box'))
        if num_bboxes == 0:
            return

        for image in image_obj.xml_root.iter('image'):
            image_name = image.attrib['name']
            image_filepath = os.path.join(image_obj.xml_dirname, image_name)

            if any(pattern in image_name.lower() for pattern in settings.INVALID_PATTERNS):
                logging.info(">>>>>> Image {} not valid to the process".format(image_name))
                continue

            if os.path.exists(image_filepath):
                image_obj.images.append(image.attrib['name'])
            else:
                logging.info(">>>>>> File image {} exist in XML file but does not exist in fs.".format(image_filepath))
                continue

            width = float(image.attrib['width'])
            height = float(image.attrib['height'])

            if image_obj.tile_image:
                img = cv2.imread(image_filepath, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

                if img is None:
                    logging.info(">>>>>> {} image filename from {} XML file, is empty".format(image_filepath,
                                                                                              image_obj.xml_filename))
                    return

                # if len(image_obj.images) == 0:
                #     logging.info(">>>>>> No image inside {} were valid!".format(image_obj.xml_filename))
                #     return

                used_tiles = self.build_rows(image_obj, image, image_filepath, width, height)
                image_obj.used_tiles_list.append(used_tiles)
                self.write_image_files(image_obj, img, width, height, used_tiles)

    def setup_output_path(self, image_obj):
        """
        :param image_obj:
        :return:
        """
        entry_path_norm = os.path.normpath(image_obj.entry_path)
        folder_name = os.path.basename(entry_path_norm)
        xml_path_splitted = os.path.normpath(os.path.dirname(image_obj.xml_filename))
        xml_path_splitted = xml_path_splitted.split(os.sep)
        index_ = xml_path_splitted.index(folder_name)
        xml_path_splitted[index_] = folder_name + '-cropped'
        os_name = platform.system()

        if os_name == "Linux":
            output_path = os.path.join("/", *xml_path_splitted)
        elif os_name == 'Windows':
            output_path = os.path.join(*xml_path_splitted)
        else:
            output_path = os.path.dirname(image_obj.xml_filename)
            logging.warning(">>>> Platform not recognized. The folder {} was not "
                            "created. Output path was set as {}".format(folder_name + '-cropped', output_path))

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def build_rows(self, image_obj, image_item, image_filename, width, height):
        """
        :param image_obj:
        :param image_item:
        :param image_filename:
        :param width:
        :param height:
        :return:
        """
        output_path = self.setup_output_path(image_obj)
        used_tiles = set()

        for box in image_item.iter('box'):
            label = box.attrib['label']
            x_tl = float(box.attrib['xtl'])
            y_tl = float(box.attrib['ytl'])
            x_br = float(box.attrib['xbr'])
            y_br = float(box.attrib['ybr'])

            if image_obj.tile_image:
                tile_start_x = int(x_tl / image_obj.tile_size) * image_obj.tile_size
                tile_start_y = int(y_tl / image_obj.tile_size) * image_obj.tile_size

                x_parts = [x_tl]
                y_parts = [y_tl]
                box_width = x_br - x_tl
                box_height = y_br - y_tl

                for w in range(tile_start_x, int(x_br + 0.5), image_obj.tile_size):
                    next_x = min(float(w + image_obj.tile_size), x_br)
                    x_parts.append(next_x)

                for h in range(tile_start_y, int(y_br + 0.5), image_obj.tile_size):
                    next_y = min(float(h + image_obj.tile_size), y_br)
                    y_parts.append(next_y)

                for h in range(len(y_parts) - 1):
                    tile_y = int(y_parts[h] / image_obj.tile_size)

                    for w in range(len(x_parts) - 1):
                        box_part_width = x_parts[w + 1] - x_parts[w]
                        box_part_height = y_parts[h + 1] - y_parts[h]
                        box_part_width_percentage = box_part_width / box_width
                        box_part_height_percentage = box_part_height / box_height

                        if box_part_width_percentage < 0.4 or box_part_height_percentage < 0.4:
                            continue

                        tile_x = int(x_parts[w] / image_obj.tile_size)

                        used_tiles.add((tile_x, tile_y))
                        tiled_filename = str(tile_x) + "_" + str(tile_y) + '.jpg'
                        filepath = os.path.join(output_path, tiled_filename)

                        xmin_norm = round((x_parts[w] - tile_x * image_obj.tile_size) / image_obj.tile_size, 9)
                        ymin_norm = round((y_parts[h] - tile_y * image_obj.tile_size) / image_obj.tile_size, 9)
                        xmax_norm = min(1.0, round(xmin_norm + box_part_width / image_obj.tile_size, 9))
                        ymax_norm = min(1.0, round(ymin_norm + box_part_height / image_obj.tile_size, 9))

                        image_obj.rows.append(['UNASSIGNED', filepath, label, xmin_norm, ymin_norm, xmax_norm, ymin_norm,
                                               xmax_norm, ymax_norm, xmin_norm, ymax_norm])
            else:
                xmin_norm = round(x_tl / width, 9)
                xmax_norm = round(x_br / width, 9)
                ymin_norm = round(y_tl / height, 9)
                ymax_norm = round(y_br / height, 9)

                image_obj.rows.append(['UNASSIGNED', image_filename, label, xmin_norm, ymin_norm, xmax_norm, ymin_norm,
                                       xmax_norm, ymax_norm, xmin_norm, ymax_norm])
        return used_tiles

    def write_image_files(self, image_obj, image, width, height, used_tiles):
        """
        :param image_obj:
        :param image:
        :param width:
        :param height:
        :param used_tiles:
        :return:
        """
        output_path = self.setup_output_path(image_obj)

        for tiles in used_tiles:
            tile_x = tiles[0]
            tile_y = tiles[1]
            tile_x_start = tile_x * image_obj.tile_size
            tile_y_start = tile_y * image_obj.tile_size

            tiled_filename = str(tile_x) + "_" + str(tile_y) + '.jpg'
            tiled_img = self.crop_image(image, tile_x_start, tile_y_start,
                                        min(tile_x_start + image_obj.tile_size, width),
                                        min(tile_y_start + image_obj.tile_size, height))
            cv2.imwrite(os.path.join(output_path, tiled_filename), tiled_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

    def crop_image(self, im, xtl, ytl, xbr, ybr):
        """
        :param im:
        :param xtl:
        :param ytl:
        :param xbr:
        :param ybr:
        :return:
        """
        roi = im[int(ytl):int(ybr), int(xtl):int(xbr)]
        return roi


