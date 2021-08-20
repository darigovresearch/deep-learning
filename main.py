import logging
import argparse

from satellite import start
from coloredlogs import ColoredFormatter

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')


def main(app, network_type, augment_data, is_training, is_predicting):
    """
    """
    if app == 'satellite':
        start.main(network_type, augment_data, is_training, is_predicting)
    elif app == 'insects':
        pass
    else:
        pass


if __name__ == '__main__':
    """
    Example:
        > python main.py -application APP -model MODEL -augment BOOLEAN -train BOOLEAN -predict BOOLEAN -verbose BOOLEAN
                
    Usage:
        > python main.py -application satellite -model unet -augment False -train True -predict False -verbose True
        > python main.py -application satellite -model unet -augment False -train False -predict True -verbose True
        > python main.py -application satellite -model unet -augment False -train True -predict True -verbose True
    """
    parser = argparse.ArgumentParser(description='Integrate some of the main Deep Learning models for remote sensing '
                                                 'image analysis and mapping')
    parser.add_argument('-application', action="store", dest='application', help='Defines which purpose the DL will '
                                                                                 'be trained: satellite or insects')
    parser.add_argument('-model', action="store", dest='model', help='Deep Learning model name: unet, deeplabv3')
    parser.add_argument('-augment', action="store", dest='augment', help='If True, augment training data')
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

    main(args.application, args.model, args.augment, args.train, args.predict)



