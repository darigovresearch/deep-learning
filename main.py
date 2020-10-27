import os
import logging
import time
import argparse
import settings
import infer

from dl.model import loader
from dl.model import helper
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

        path_train_images = os.path.join(load_param['image_training_folder'], 'image')
        path_train_labels = os.path.join(load_param['image_training_folder'], 'label')
        path_val_images = os.path.join(load_param['image_validation_folder'], 'image')
        path_val_labels = os.path.join(load_param['image_validation_folder'], 'label')

        train_images = loader.Loader(path_train_images)
        train_labels = loader.Loader(path_train_labels)
        val_images = loader.Loader(path_val_images)
        val_labels = loader.Loader(path_val_labels)

        batch_size = load_param['batch_size']
        img_size = (load_param['input_size_w'], load_param['input_size_h'])

        train_generator_obj = helper.Helper(batch_size, img_size, train_images.get_list_images(),
                                            train_labels.get_list_images())
        val_generator_obj = helper.Helper(batch_size, img_size, val_images.get_list_images(),
                                          val_labels.get_list_images())

        history = dl_obj.get_model().fit(train_generator_obj,
                                         steps_per_epoch=train_generator_obj.__len__(),
                                         validation_data=val_generator_obj,
                                         validation_steps=val_generator_obj.__len__(),
                                         epochs=load_param['epochs'],
                                         callbacks=dl_obj.get_callbacks())

    if eval(is_predicting):
        dl_obj = get_dl_model(network_type, load_param, True, False)
        infer.Infer().predict_deep_network(dl_obj, load_param)

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



