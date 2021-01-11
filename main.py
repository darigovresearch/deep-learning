import os
import logging
import time
import json
import random
import argparse
import settings
import tensorflow as tf

from datetime import datetime
from output import infer
from input import loader, helper
from dl.model import unet
from coloredlogs import ColoredFormatter

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def get_dl_model(network_type, load_param, is_pretrained, is_saved):
    """
    Setup input size and compile the model architecture, according to network_type

    :param network_type: Deep Learning architecture: unet, deeplabv3, so on
    :param load_param: The parameters from the network specified. The parameters are pre-stablished in settings.py file
    :param is_pretrained: a boolean, if True, the model will be compiled with a pre-trained weights, also
    established in settings.py file [output_checkpoints]
    :param is_saved: a boolean, if True, the model after built, will be saved in a pre-stablished path,
    set in settings.py file [save_model_dir]
    :return: a compiled keras model

    Source:
        - https://github.com/divamgupta/image-segmentation-keras
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
        batch_size = load_param['batch_size']
        img_size = (load_param['input_size_w'], load_param['input_size_h'])

        tf.keras.backend.clear_session()
        dl_obj = get_dl_model(network_type, load_param, False, True)

        logging.info(">> Loading input datasets...")

        path_train_images = os.path.join(load_param['image_training_folder'], 'image')
        path_train_labels = os.path.join(load_param['image_training_folder'], 'label')
        train_images_paths = loader.Loader(path_train_images)
        train_labels_paths = loader.Loader(path_train_labels)
        train_images_paths = train_images_paths.get_list_images()
        train_labels_paths = train_labels_paths.get_list_images()

        train_list = list(zip(train_images_paths, train_labels_paths))
        random.shuffle(train_list)
        train_images_paths, train_labels_paths = zip(*train_list)

        percent_val = int(len(train_images_paths) * settings.VALIDATION_SPLIT)
        train_input_images = train_images_paths[:-percent_val]
        train_input_labels = train_labels_paths[:-percent_val]
        val_input_images = train_images_paths[-percent_val:]
        val_labels_images = train_labels_paths[-percent_val:]

        train_generator_obj = helper.Helper(batch_size, img_size, train_input_images, train_input_labels)
        val_generator_obj = helper.Helper(batch_size, img_size, val_input_images, val_labels_images)

        dl_obj.get_model().compile(optimizer=dl_obj.get_optimizer(), loss=dl_obj.get_loss(), metrics=['accuracy'])

        history = dl_obj.get_model().fit(train_generator_obj,
                                         epochs=load_param['epochs'],
                                         validation_data=val_generator_obj,
                                         callbacks=dl_obj.get_callbacks(),
                                         steps_per_epoch=train_generator_obj.__len__(),
                                         validation_steps=val_generator_obj.__len__())

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



