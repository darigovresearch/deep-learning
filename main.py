import os
import logging
import time
import argparse
import settings
import imageio
import cv2
import numpy as np
import tifffile as tiff

from dl.model import utils
from dl.model import unet
from coloredlogs import ColoredFormatter

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_dl_model(network_type, load_param, is_pretrained, is_saved):
    """
    Source: https://github.com/divamgupta/image-segmentation-keras
    :param network_type:
    :param load_param:
    :param is_pretrained:
    :param is_saved:
    :return:
    """
    model_obj = None

    if network_type == 'unet':
        logging.info(">> UNET model selected...")

        input_size = (load_param['input_size_w'], load_param['input_size_h'], load_param['input_size_c'])

        model_obj = unet.UNet(input_size, is_pretrained, is_saved)

    # TODO: include deeplabv3 as alternative to the set of dl models
    elif network_type == 'deeplabv3':
        logging.info(">> DEEPLABv3 model selected...")
        pass

    else:
        logging.info("The type of neural network [{}], does not exist. Pick "
                     "another and try it again!". format(network_type))

    return model_obj


def main(network_type, is_training, is_predicting):
    """
    :param network_type:
    :param is_training:
    :param is_predicting:

    validation_steps and steps_per_epoch: https://stackoverflow.com/questions/51885739/
                                          how-to-properly-set-steps-per-epoch-and-validation-steps-in-keras
    """
    start_time = time.time()
    logging.info("Starting process...")

    load_param = settings.DL_PARAM[network_type]

    if eval(is_training):
        dl_obj = get_dl_model(network_type, load_param, False, False)
        path_train_samples = os.path.join(settings.DL_PARAM[network_type]['image_training_folder'], 'image')
        path_val_samples = os.path.join(settings.DL_PARAM[network_type]['image_validation_folder'], 'image')
        num_train_samples = len(os.listdir(path_train_samples))
        num_val_samples = len(os.listdir(path_val_samples))

        train_generator_obj, val_generator_obj = utils.DL().training_generator(network_type, True)
        dl_obj.get_model().fit(train_generator_obj,
                               steps_per_epoch=np.ceil(num_train_samples //
                                                       settings.DL_PARAM[network_type]['batch_size']),
                               validation_data=val_generator_obj,
                               validation_steps=int(num_val_samples //
                                                    settings.DL_PARAM[network_type]['batch_size']),
                               epochs=settings.DL_PARAM[network_type]['epochs'],
                               callbacks=dl_obj.get_callbacks())

    if eval(is_predicting):
        dl_obj = get_dl_model(network_type, load_param, True, False)

        list_images_to_predict = os.listdir(load_param['image_prediction_folder'])
        list_images_to_predict = [file for file in list_images_to_predict if file.endswith(settings.VALID_PREDICTION_EXTENSION)]

        classes = load_param['classes']
        COLOR_DICT = np.array(['nut', 'palm', 'other'])

        for item in list_images_to_predict:
            complete_path = os.path.join(settings.DL_PARAM[network_type]['image_prediction_folder'], item)
            filename, extension = os.path.splitext(item)

            if extension.lower() == ".tif" or extension.lower() == ".tiff":
                image_full = tiff.imread(complete_path)
                dims = image_full.shape
            else:
                image_full = cv2.imread(complete_path)
                dims = image_full.shape

            # img = image_full / 255
            # img = np.reshape(image_full, (1, 256, 256, 3))
            # y_prob = dl_obj.get_model().predict(img)
            # y_classes = y_prob.argmax(axis=-1)
            # output = np.reshape(y_classes, (256, 256))

            pr = dl_obj.get_model().predict(np.array([image_full]))[0]
            pr = pr.reshape((256, 256, 3)).argmax(axis=2)
            pred_mask = np.argmax(pr, axis=-1)
            output = np.reshape(pred_mask, (256, 256))

            img_color = np.zeros((256, 256, 3), dtype='uint8')
            for j in range(dims[0]):
                for i in range(dims[1]):
                    img_color[j, i] = classes[COLOR_DICT[output[j, i]]]

            prediction_path = os.path.join(settings.DL_PARAM[network_type]['output_prediction'], filename + '.png')
            imageio.imwrite(prediction_path, img_color)

            # TODO:
            #  1. save prediction in png
            #  2. merge the prediction if it was sliced
            #  3. poligonize the merged prediction image

    end_time = time.time()
    logging.info("Whole process completed! [Time: {0:.5f} seconds]!".format(end_time-start_time))


if __name__ == '__main__':
    """
    usage:
        python main.py -model unet -train True -predict False -verbose True
        python main.py -model unet -train False -predict True -verbose True
        python main.py -model unet -train True -predict True -verbose True
    """
    parser = argparse.ArgumentParser(
        description='Make a stack composition from Sentinel-1 polarization bands, which enhances '
                    'land-changes under the canopies')
    parser.add_argument('-model', action="store", dest='model', help='Deep Learning model name: unet, deeplabv3')
    parser.add_argument('-train', action="store", dest='train', help='Perform neural network training?')
    parser.add_argument('-predict', action="store", dest='predict', help='Perform neural network prediction?')
    parser.add_argument('-verbose', action="store", dest='verbose', help='Print log of processing')
    args = parser.parse_args()

    if eval(args.verbose):
        log = logging.getLogger('')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cf = ColoredFormatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ")
        ch.setFormatter(cf)
        log.addHandler(ch)

        fh = logging.FileHandler('logging.log')
        fh.setLevel(logging.INFO)
        ff = logging.Formatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ",
                               datefmt='%Y.%m.%d %H:%M:%S')
        fh.setFormatter(ff)
        log.addHandler(fh)

        # log.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")

    main(args.model, args.train, args.predict)



