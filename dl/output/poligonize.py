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
    """
    def __init__(self):
        pass

    def create_geometry(self, corners, hierarchy, image):
        """
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

    def get_classes_gt(self, complete_path_png):
        """
        """
        logging.info(">>>>>> Checking classes presenting on images according to the ones of interest...")

        image_segmented = cv2.imread(complete_path_png)
        image_segmented = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB)
        height, width, bands = image_segmented.shape

        gt_classes = []
        for key, value in load_param['classes'].items():
            for i in range(height):
                for j in range(width):
                    if (image_segmented[i, j][0] == value[0]) and (image_segmented[i, j][1] == value[1]) and \
                            (image_segmented[i, j][2] == value[2]) and (key not in gt_classes):
                        gt_classes.append(key)

        return gt_classes

    def get_image_by_class(self, complete_path_png, key):
        """
        :param complete_path_png:
        :param key:
        :return:
        """
        logging.info(">>>>>> Spliting images by classes...")

        image_segmented = cv2.imread(complete_path_png)
        image_segmented = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB)

        value = load_param['classes'][key]
        height, width, bands = image_segmented.shape

        for i in range(height):
            for j in range(width):
                if not ((image_segmented[i, j][0] == value[0]) and (image_segmented[i, j][1] == value[1]) and
                        (image_segmented[i, j][2] == value[2])):
                    image_segmented[i, j][0] = 0
                    image_segmented[i, j][1] = 0
                    image_segmented[i, j][2] = 0

        return image_segmented

    def create_shapefile(self, segmented, complete_path_png, complete_path_vector, vector_type='ESRI Shapefile'):
        logging.info(">>>>>> Creating vector file...")

        filename = basename(complete_path_png)
        name = os.path.splitext(filename)[0]

        image = gdal.Open(complete_path_png)

        driver = ogr.GetDriverByName(vector_type)
        ds = driver.CreateDataSource(complete_path_vector)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(image.GetProjection())

        _area = ogr.FieldDefn('area', ogr.OFTReal)
        _class = ogr.FieldDefn('class', ogr.OFTString)

        gt_classes = self.get_classes_gt(segmented)

        classes_and_geometries = {}
        for k in range(len(gt_classes)):
            image_segmented = self.get_image_by_class(segmented, gt_classes[k])
            image_segmented_ingray = cv2.cvtColor(image_segmented, cv2.COLOR_RGB2GRAY)

            thresh = cv2.threshold(image_segmented_ingray, 127, 255, 0)[1]
            im2, corners, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            geometries = self.create_geometry(corners, hierarchy, image)

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

    def polygonize(self, segmented, image, output):
        logging.info(">>>> Initiating polygonization of the raster result...")

        ext = os.path.splitext(segmented)[1]

        if ext.lower() not in settings.VALID_PREDICTION_EXTENSION:
            logging.info(">>>>>> Image with no accept extension {}!".format(ext))

        if not os.path.isfile(segmented):
            logging.info(">>>>>> There is no corresponding PNG image for {}!".format(image))
            return

        self.create_shapefile(segmented, image, output)
