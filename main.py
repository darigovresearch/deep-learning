import os
import sys
import logging
import time
import numpy
import argparse
import settings
import tensorflow as tf

from dl.model import utils
from dl.model import unet
from coloredlogs import ColoredFormatter


def predict_deep_network(model, path_in, path_out, path_chp):
    """
    """
    logging.info(">> Perform prediction...")
    file_list = []

    for path in os.listdir(path_in):
        full_path = os.path.join(path_in, path)

        if os.path.isfile(full_path):
            file_list.append(full_path)

    model.predict_multiple(checkpoints_path=path_chp, inps=file_list,  out_dir=path_out)


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
        model_obj = unet.UNet().model_3(input_size)

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
        filepath = os.path.join(settings.DL_PARAM[network_type]['output_checkpoints'],
                                "model-{epoch:02d}.hdf5")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_accuracy',
                                               verbose=1, save_best_only=False, mode='auto'),
            tf.keras.callbacks.TensorBoard(log_dir=settings.DL_PARAM[network_type]['tensorboard_log_dir']),
        ]

        dl_obj.fit(train_generator_obj,
                   steps_per_epoch=numpy.ceil(num_train_samples / settings.DL_PARAM[network_type]['batch_size']),
                   epochs=settings.DL_PARAM[network_type]['epochs'],
                   callbacks=callbacks)

    # TODO: include inference procedures
    if eval(is_predicting):
        pass

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



