import os
import logging
import time
import argparse
import settings
import imageio
import numpy as np
import tifffile as tiff

from dl.model import utils
from dl.model import unet
from coloredlogs import ColoredFormatter

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_dl_model(network_type, load_param):
    """
    Source: https://github.com/divamgupta/image-segmentation-keras
    :param network_type:
    :param load_param:
    :return:
    """
    model_obj = None

    if network_type == 'unet':
        logging.info(">> UNET model selected...")

        input_size = (load_param['input_size_w'], load_param['input_size_h'], load_param['input_size_c'])
        num_classes = len(load_param['classes'])
        num_channels = load_param['input_size_c']

        model_obj = unet.UNet(input_size, num_classes, num_channels, False, False)

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
    dl_obj = get_dl_model(network_type, load_param)

    if eval(is_training):
        path_train_samples = os.path.join(settings.DL_PARAM[network_type]['image_training_folder'], 'image')
        num_train_samples = len(os.listdir(path_train_samples))

        train_generator_obj = utils.DL().training_generator(network_type, True)

        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        dl_obj.get_model().fit(train_generator_obj,
                               steps_per_epoch=np.ceil(num_train_samples / settings.DL_PARAM[network_type]['batch_size']),
                               epochs=settings.DL_PARAM[network_type]['epochs'],
                               callbacks=dl_obj.get_callbacks())

    if eval(is_predicting):
        list_images_to_predict = os.listdir(settings.DL_PARAM[network_type]['image_prediction_folder'])
        list_images_to_predict = [file for file in list_images_to_predict if file.endswith(settings.VALID_PREDICTION_EXTENSION)]

        for item in list_images_to_predict:
            complete_path = os.path.join(settings.DL_PARAM[network_type]['image_prediction_folder'], item)

            # TODO: do all tests if entry is ok after open: dimension, type, encoding, so on
            # img = image.load_img(complete_path, target_size=(256, 256))
            # x = image.img_to_array(img)
            #
            # pr = dl_obj.predict(np.array([x]))[0]

            # pil_img = Image.fromarray((pr * 255).astype(np.uint8))
            # pil_img.save(os.path.join(settings.DL_PARAM[network_type]['output_prediction'], 'test-inference.png'))

            extension = os.path.splitext(item)[1]

            if extension.lower() == ".tif" or extension.lower() == ".tiff":
                image_full = tiff.imread(complete_path)
                # img = image.load_img(complete_path, target_size=(image_full.shape[0], image_full.shape[1]))
                # x = image.img_to_array(img)
            # else:
                # image_full = scipy.misc.imread(complete_path)

            pr = dl_obj.predict(np.array([image_full]))[0]
            pred_mask = tf.argmax(pr, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]
            imageio.imwrite(os.path.join(settings.DL_PARAM[network_type]['output_prediction'], 'test-inference.png'),
                            pred_mask)

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

        log.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")

    main(args.model, args.train, args.predict)



