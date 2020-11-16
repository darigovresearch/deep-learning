import os
import gdal
import logging
import cv2
import osgeo.ogr as ogr
import osgeo.osr as osr
import settings

from os.path import basename


class Poligonize:
    """
    Perform poligonization [raster to vector] operations, despite the utils methods for it
    """
    def __init__(self):
        pass

    def _create_geometry(self, corners, hierarchy, image):
        """
        Interpret the corners and its hierarchy from findContours openCV method, and convert it in a geographic
        geometry, according to the image metadata

        :param corners: the opencv findContours outcomes
        :param hierarchy: the hierarchy of contours found: sometimes, there is "holes" inside a polygon
        :param image: the correspondent geographic format [raster] of the segmented image
        :return: the geographic vector features/geometries of the contours found in raster format
        """
        logging.info(">>>>>> Creating geometries...")

        gt = image.GetGeoTransform()
        poly = ogr.Geometry(ogr.wkbPolygon)

        geom = []
        for i in range(len(corners)):
            flag = True
            ring = ogr.Geometry(ogr.wkbLinearRing)

            for coord in corners[i]:
                x_geo = gt[0] + coord[0][0] * gt[1] + coord[0][1] * gt[2]
                y_geo = gt[3] + coord[0][0] * gt[4] + coord[0][1] * gt[5]
                ring.AddPoint(x_geo, y_geo)

                if flag is True:
                    flag = False
                    initial_x = x_geo
                    initial_y = y_geo

            ring.AddPoint(initial_x, initial_y)
            poly.AddGeometry(ring)

            # TODO: refatorar condicionais
            if i+1 < len(corners):
                if (hierarchy[0, i, 2] != -1) or ((hierarchy[0, i, 2] == -1) and (hierarchy[0, i, 3] != -1) and
                                                  (hierarchy[0, i+1, 3] != -1)):
                    continue
                else:
                    geom.append(ogr.CreateGeometryFromWkt(poly.ExportToWkt()))
                    poly = ogr.Geometry(ogr.wkbPolygon)
            else:
                geom.append(ogr.CreateGeometryFromWkt(poly.ExportToWkt()))
                poly = ogr.Geometry(ogr.wkbPolygon)

        return geom

    def get_classes_gt(self, segmented, classes):
        """
        Read all pixels and check if there is more classes than the ones specified in classes dict

        :param segmented: absolute path of the image segmentation/classification result. A raster
                          [JPG, PNG, TIFF, so on] version from the inference operation
        :param classes: the list of classes and respectively colors
        :return: a list with all classes presented in the image (i.e. only if within classes dict as well)
        """
        logging.info(">>>>>> Checking classes presenting on images according to the ones of interest...")

        image_segmented = cv2.imread(segmented)
        image_segmented = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB)
        height, width, bands = image_segmented.shape

        gt_classes = []
        for key, value in classes.items():
            for i in range(height):
                for j in range(width):
                    if (image_segmented[i, j][0] == value[0]) and (image_segmented[i, j][1] == value[1]) and \
                            (image_segmented[i, j][2] == value[2]) and (key not in gt_classes):
                        gt_classes.append(key)

        return gt_classes

    def get_image_by_class(self, segmented, classes, key):
        """
        In some cases, the segmented image might present variations classes's color, which could led in a wrong reading
        of the quantity of classes presented in the segmentation. Thus, According to the classes dict, the method
        returns a new image, with a filtered color classes

        :param segmented: absolute path of the image segmentation/classification result. A raster
                          [JPG, PNG, TIFF, so on] version from the inference operation
        :param classes: the list of classes and respectively colors
        :param key: the string class name
        :return: a new image, with a filtered color classes
        """
        logging.info(">>>>>> Spliting images by classes...")

        image_segmented = cv2.imread(segmented)
        image_segmented = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB)

        value = classes[key]
        height, width, bands = image_segmented.shape

        for i in range(height):
            for j in range(width):
                if not ((image_segmented[i, j][0] == value[0]) and (image_segmented[i, j][1] == value[1]) and
                        (image_segmented[i, j][2] == value[2])):
                    image_segmented[i, j][0] = 0
                    image_segmented[i, j][1] = 0
                    image_segmented[i, j][2] = 0

        return image_segmented

    def create_shapefile(self, segmented, classes, original_image_path, output_vector_path, vector_type='ESRI Shapefile'):
        """
        Perform operations to read original image [remote sensing data], and from its metadata, create a new vector
        file according to the classes specified in settings.py file

        :param segmented: absolute path of the image segmentation/classification result. A raster
                          [JPG, PNG, TIFF, so on] version from the inference operation
        :param classes: the list of classes and respectively colors
        :param original_image_path: the original raster image path, where the geographic metadata is
                                    read and transfer to the output
        :param output_vector_path: the output path, where the new geographic format is saved
        :param vector_type: default value is ESRI Shapefile (most commom), but GeoJSON is accepted
        """
        logging.info(">>>>>> Creating vector file...")

        filename = basename(original_image_path)
        name = os.path.splitext(filename)[0]

        image = gdal.Open(original_image_path)

        driver = ogr.GetDriverByName(vector_type)
        ds = driver.CreateDataSource(output_vector_path)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(image.GetProjection())

        _area = ogr.FieldDefn('area', ogr.OFTReal)
        _class = ogr.FieldDefn('class', ogr.OFTString)

        gt_classes = self.get_classes_gt(segmented, classes)

        classes_and_geometries = {}
        for k in range(len(gt_classes)):
            image_segmented = self.get_image_by_class(segmented, classes, gt_classes[k])
            image_segmented_ingray = cv2.cvtColor(image_segmented, cv2.COLOR_RGB2GRAY)

            thresh = cv2.threshold(image_segmented_ingray, 127, 255, 0)[1]
            im2, corners, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            geometries = self._create_geometry(corners, hierarchy, image)

            classes_and_geometries[gt_classes[k]] = geometries

        layer = ds.CreateLayer(name, srs, ogr.wkbPolygon)
        layer.CreateField(_area)
        layer.CreateField(_class)

        for key, value in classes_and_geometries.items():
            for g in range(len(value)):
                feature_defn = layer.GetLayerDefn()
                feature = ogr.Feature(feature_defn)

                area = value[g].GetArea()
                feature.SetGeometry(value[g])
                feature.SetField('area', area)
                feature.SetField('class', key)

                layer.CreateFeature(feature)

        logging.info(">>>>>> Vector file of image {} created!".format(filename))

    def polygonize(self, segmented, classes, original_image_path, output_vector_path):
        """
        Turn a JPG, PNG images in a geographic format, such as ESRI Shapefile or GeoJSON. The image must to be
        in the exact colors specified in settings.py - DL_PARAM['classes']

        :param segmented: the segmented multiclass image path
        :param classes: the list of classes and respectively colors
        :param original_image_path: the original raster image path, where the geographic metadata is read and transfer to the output
        :param output_vector_path: the output path, where the new geographic format is saved
        """
        logging.info(">>>> Initiating polygonization of the raster result...")

        ext = os.path.splitext(segmented)[1]

        if ext.lower() not in settings.VALID_PREDICTION_EXTENSION:
            logging.info(">>>>>> Image with no accept extension {}!".format(ext))

        if not os.path.isfile(segmented):
            logging.info(">>>>>> There is no corresponding PNG image for {}!".format(original_image_path))
            return

        self.create_shapefile(segmented, classes, original_image_path, output_vector_path, 'ESRI Shapefile')
