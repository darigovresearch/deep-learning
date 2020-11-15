import os
import logging
import time
import json
import argparse
import settings

from datetime import datetime
from dl.output import infer
from dl.input import helper, loader
from dl.model import unet

from coloredlogs import ColoredFormatter

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_dl_model(network_type, load_param, is_pretrained, is_saved):
    """
    Setup input size and compile the model architecture, according to network_type

    :param network_type: Deep Learning architecture: unet, deeplabv3, so on
    :param load_param: The parameters from the network specified. The parameters are pre-stablished in settings.py file
    :param is_pretrained: a boolean, if True, the model will be compiled with a pre-trained weights, also
    stablished in settings.py file [output_checkpoints]
    :param is_saved: a boolean, if True, the model after built, will be saved in a pre-stablished path,
    set in settings.py file [save_model_dir]
    :return: a compiled keras model

    Source:
    https://github.com/divamgupta/image-segmentation-keras
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
    Initiate the processing: execute training and predictions according to is_training and is_predicting variables
    :param network_type: Deep Learning architecture: unet, deeplabv3, so on
    :param is_training: a boolean, if True, the input samples are then loaded, the specified model is compiled and
    the training is performed
    :param is_predicting: a boolean, if True, the deep learning inferences is performed according
    to the model specified

    Sources:
        - validation_steps and steps_per_epoch: https://stackoverflow.com/questions/51885739/
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

        timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M")
        history_file = os.path.join(load_param['output_history'], "history-" + str(timestamp) + ".json")
        with open(history_file, 'w') as f:
            json.dump(history.history, f)

    if eval(is_predicting):
        dl_obj = get_dl_model(network_type, load_param, True, False)
        infer.Infer().predict_deep_network(dl_obj, load_param)

    end_time = time.time()
    logging.info("Whole process completed! [Time: {0:.5f} seconds]!".format(end_time-start_time))


if __name__ == '__main__':
    """
    Example:
        > python main.py -model MODEL -train BOOLEAN -predict BOOLEAN -verbose BOOLEAN
                
    Usage:
        > python main.py -model unet -train True -predict False -verbose True
        > python main.py -model unet -train False -predict True -verbose True
        > python main.py -model unet -train True -predict True -verbose True
    """
    parser = argparse.ArgumentParser(description='Integrate some of the main Deep Learning models for remote sensing '
                                                 'image analysis and mapping')
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



