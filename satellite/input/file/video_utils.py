import os
import logging
import argparse
import platform

from cv2 import cv2


class VideoUtils:
    """ Utils methods to deal with video cvat xml preprocessing """

    def __init__(self, video_obj):
        self.build_xml_video_object(video_obj)

    def build_xml_video_object(self, video_obj):
        """
        :return:
        """
        if not os.path.exists(video_obj.video_filename):
            logging.info(">>>> Video file {} exist in XML file but does not exist in fs.".
                         format(video_obj.video_filename))
            return

        width = video_obj.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_obj.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        rotation = video_obj.vidcap.get(cv2.CAP_PROP_ORIENTATION_META)

        task = video_obj.xml_root[1].find('task')
        original_size = task.find('original_size')
        width_from_xml = float(original_size.find('width').text)
        height_from_xml = float(original_size.find('height').text)

        if width_from_xml != width or height_from_xml != height:
            if width_from_xml != height or height_from_xml != width:
                logging.info(">>>> Different image/video dimension from file to xml metadata. Check it!")
                return
            width = width_from_xml
            height = height_from_xml

        for track in video_obj.xml_root.iter('track'):
            self.build_rows(video_obj, track, width, height)

        self.write_frame_image_files(video_obj, width, height, rotation)
        self.delete_rows_that_file_does_not_exist(video_obj)

    def setup_output_path(self, video_obj):
        """
        :param video_obj:
        :return:
        """
        entry_path_norm = os.path.normpath(video_obj.entry_path)
        folder_name = os.path.basename(entry_path_norm)
        xml_path_splitted = os.path.normpath(os.path.dirname(video_obj.xml_filename))
        xml_path_splitted = xml_path_splitted.split(os.sep)
        index_ = xml_path_splitted.index(folder_name)
        xml_path_splitted[index_] = folder_name + '-cropped'
        os_name = platform.system()

        if os_name == "Linux":
            output_path = os.path.join("/", *xml_path_splitted)
        elif os_name == 'Windows':
            output_path = os.path.join(*xml_path_splitted)
        else:
            output_path = os.path.dirname(video_obj.xml_filename)
            logging.warning(">>>> Platform not recognized. The folder {} was not "
                            "created. Output path was set as {}".format(folder_name + '-cropped', output_path))

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def build_rows(self, video_obj, track, width, height):
        """
        :param video_obj:
        :param track:
        :param width:
        :param height:
        :return:
        """
        output_path = self.setup_output_path(video_obj)

        track_id = track.attrib['id']
        label = track.attrib['label']

        for box in track.iter('box'):
            outside = self.str2bool(box.attrib['outside'])

            if outside:
                continue

            frame_num = int(box.attrib['frame'])
            x_tl = float(box.attrib['xtl'])
            y_tl = float(box.attrib['ytl'])
            x_br = float(box.attrib['xbr'])
            y_br = float(box.attrib['ybr'])

            video_obj.video_labels[frame_num].append([track_id, label, x_tl, y_tl, x_br, y_br])

            if video_obj.tile_image:
                tile_start_x = int(x_tl / video_obj.tile_size) * video_obj.tile_size
                tile_start_y = int(y_tl / video_obj.tile_size) * video_obj.tile_size

                x_parts = [x_tl]
                y_parts = [y_tl]
                box_width = x_br - x_tl
                box_height = y_br - y_tl

                for w in range(tile_start_x, int(x_br + 0.5), video_obj.tile_size):
                    next_x = min(w + video_obj.tile_size, x_br)
                    x_parts.append(next_x)

                for h in range(tile_start_y, int(y_br + 0.5), video_obj.tile_size):
                    next_y = min(h + video_obj.tile_size, y_br)
                    y_parts.append(next_y)

                for h in range(len(y_parts) - 1):
                    tile_y = int(y_parts[h] / video_obj.tile_size)

                    for w in range(len(x_parts) - 1):
                        box_part_width = x_parts[w + 1] - x_parts[w]
                        box_part_height = y_parts[h + 1] - y_parts[h]
                        box_part_width_percentage = box_part_width / box_width
                        box_part_height_percentage = box_part_height / box_height

                        if box_part_width_percentage < 0.4 or box_part_height_percentage < 0.4:
                            continue

                        tile_x = int(x_parts[w] / video_obj.tile_size)
                        video_obj.used_tiles[frame_num].add((tile_x, tile_y))

                        tiled_filename = 'Frame' + str(frame_num) + '_' + str(tile_x) + "_" + \
                                         str(tile_y) + '.jpg'
                        filepath = os.path.join(output_path, tiled_filename)

                        xmin_norm = round((x_parts[w] - tile_x * video_obj.tile_size) / video_obj.tile_size, 9)
                        ymin_norm = round((y_parts[h] - tile_y * video_obj.tile_size) / video_obj.tile_size, 9)
                        xmax_norm = min(1.0, round(xmin_norm + box_part_width / video_obj.tile_size, 9))
                        ymax_norm = min(1.0, round(ymin_norm + box_part_height / video_obj.tile_size, 9))

                        video_obj.rows.append(
                            ['UNASSIGNED', filepath, label, xmin_norm, ymin_norm, xmax_norm, ymin_norm,
                             xmax_norm, ymax_norm, xmin_norm, ymax_norm])
            else:
                frame_filename = 'Frame' + str(frame_num) + '.jpg'
                filepath = os.path.join(output_path, frame_filename)

                xmin_norm = round(x_tl / width, 9)
                xmax_norm = round(x_br / width, 9)
                ymin_norm = round(y_tl / height, 9)
                ymax_norm = round(y_br / height, 9)

                video_obj.rows.append(
                    ['UNASSIGNED', filepath, label, xmin_norm, ymin_norm, xmax_norm, ymin_norm,
                     xmax_norm, ymax_norm, xmin_norm, ymax_norm])

    def write_frame_image_files(self, video_obj, width, height, rotation):
        """
        :param video_obj:
        :param width:
        :param height:
        :param rotation:
        :return:
        """
        output_path = self.setup_output_path(video_obj)

        for frame_num in range(len(video_obj.video_labels)):
            if len(video_obj.used_tiles[frame_num]) == 0:
                continue

            read_flag, frame = self.get_frame_from_video(video_obj.vidcap, frame_num)
            if not read_flag:
                continue

            if rotation == 0:
                pass
            elif rotation == -90.0:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180.0:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 90.0:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                logging.error(">>>>>> Failed during rotation video reading!")
                break

            if video_obj.tile_image:
                for tiles in video_obj.used_tiles[frame_num]:
                    tile_x = tiles[0]
                    tile_y = tiles[1]
                    tile_x_start = tile_x * video_obj.tile_size
                    tile_y_start = tile_y * video_obj.tile_size

                    tiled_filename = 'Frame' + str(frame_num) + '_' + str(tile_x) + "_" + str(tile_y) + '.jpg'
                    filepath = os.path.join(output_path, tiled_filename)
                    tiled_img = self.crop_image(frame, tile_x_start, tile_y_start,
                                                min(tile_x_start + video_obj.tile_size, width),
                                                min(tile_y_start + video_obj.tile_size, height))
                    cv2.imwrite(filepath, tiled_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                if len(video_obj.video_labels[frame_num]) > 0:
                    frame_filename = 'Frame' + str(frame_num) + '.jpg'
                    filepath = os.path.join(output_path, frame_filename)
                    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        video_obj.vidcap.release()

    def delete_rows_that_file_does_not_exist(self, video_obj):
        """
        :param video_obj:
        :return:
        """
        video_obj.rows = list(filter(lambda row: os.path.exists(row[1]), video_obj.rows))

    def str2bool(self, v):
        """
        :param v:
        :return:
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

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

    def get_frame_from_video(self, vidcap, frame_number):
        """
        :param vidcap:
        :param frame_number:
        :return:
        """
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        read_flag, frame = vidcap.read()

        return read_flag, frame


