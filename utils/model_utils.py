import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf

from cv2 import cv2
from satellite import settings
from coloredlogs import ColoredFormatter


def decode_fn(record_bytes):
    """
    :param record_bytes:
    :return:
    """
    example = tf.io.parse_single_example(
                    record_bytes,
                    {"image/height": tf.io.FixedLenFeature([], dtype=tf.int64),
                     "image/width": tf.io.FixedLenFeature([], dtype=tf.int64),
                     'image/filename': tf.io.FixedLenFeature([], dtype=tf.string),
                     'image/source_id': tf.io.FixedLenFeature([], dtype=tf.string),
                     'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
                     'image/format': tf.io.FixedLenFeature([], dtype=tf.string),
                     'image/object/bbox/xmin': tf.io.FixedLenFeature([], dtype=tf.float32),
                     'image/object/bbox/xmax': tf.io.FixedLenFeature([], dtype=tf.float32),
                     'image/object/bbox/ymin': tf.io.FixedLenFeature([], dtype=tf.float32),
                     'image/object/bbox/ymax': tf.io.FixedLenFeature([], dtype=tf.float32),
                     'image/object/class/text': tf.io.FixedLenFeature([], dtype=tf.string),
                     'image/object/class/label': tf.io.FixedLenFeature([], dtype=tf.int64)}
                )
    return example


def representative_dataset(image_path):
    """
    Create image sample using tfrecord entry

    :param image_path: Absolute path to TFRecord
    :return: yield image in uint8 format
    """
    dataset = tf.data.TFRecordDataset(image_path)
    dataset = dataset.map(decode_fn)

    for data in dataset.take(settings.NUM_IMAGES):
        image_raw = tf.io.decode_raw(data['image/encoded'], tf.uint8)
        width = data['image/width'].numpy()
        height = data['image/height'].numpy()

        image = cv2.cvtColor(image_raw.numpy(), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image = image.astype("uint8")
        image = np.expand_dims(image, axis=1)
        image = image.reshape(1, settings.TILE_SIZE, settings.TILE_SIZE, 3)
        yield [image.astype("uint8")]


def tfmodel2tflite(tfmodel_path, tflite_output, sample_images, keras_model=False):
    """
    Convert TFmodel in TFlite

    "For full integer quantization, you need to calibrate or estimate the range, i.e, (min, max) of all
    floating-point tensors in the model. Unlike constant tensors such as weights and biases, variable
    tensors such as model input, activations (outputs of intermediate layers) and model output cannot
    be calibrated unless we run a few inference cycles. As a result, the converter requires a representative
    dataset to calibrate them. This dataset can be a small subset (around ~100-500 samples) of the
    training or validation data"

    Source:
        - https://www.tensorflow.org/lite/performance/post_training_quantization#full_
          integer_quantization_of_weights_and_activations

    :param tfmodel_path:
    :param tflite_output:
    :param sample_images: Absolute path to test.records, used to representative datasets
    :param keras_model:
    """
    model_name = tfmodel_path.split(os.sep)[-2]
    tflite_output_filename = os.path.join(tflite_output, model_name)

    os.mkdir(tflite_output_filename)
    tflite_output_filename = os.path.join(tflite_output_filename, 'model.tflite')

    logging.info(">>>> Converting TF model {} in TFLite...".format(tfmodel_path))
    model = tf.saved_model.load(tfmodel_path)
    model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[0].set_shape([1, settings.TILE_SIZE,
                                                                                            settings.TILE_SIZE, 3])

    if keras_model:
        converter = tf.lite.TFLiteConverter.from_keras_model(tfmodel_path)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=tfmodel_path,
                                                             signature_keys=['serving_default'])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.experimental_new_quantizer = True

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_dataset(sample_images)
    tflite_model = converter.convert()

    with open(tflite_output_filename, 'wb') as f:
        f.write(tflite_model)

    logging.info(">>>> Model {} TFLite saved!".format(tflite_output))


def main(arguments):
    """
    DESCRIPTION

    :param arguments: User parameters
    """
    tfmodel_path = arguments.tfmodel_path
    sample_images = arguments.sample_image
    is_keras = eval(args.keras_model)
    tflite_output = arguments.tflite_output

    tfmodel2tflite(tfmodel_path, tflite_output, sample_images, is_keras)


# TODO: DESCRIPTION
if __name__ == '__main__':
    """    
    DESCRIPTION

    Optional arguments:
      -h, --help             Show this help message and exit
      -t, -tfmodel_path      DESCRIPTION
      -k, -keras_model       DESCRIPTION
      -l, -tflite_output     DESCRIPTION
      -s, -sample_image      DESCRIPTION      
      -v, -verbose           Boolean to print output logging or not
     
     Usage: 
        > python model_utils.py [-h] [-t TFMODEL_PATH] [-k KERAS_MODEL] [-l TFLITE_OUTPUT] 
                                [-s SAMPLE_IMAGE] [-v VERBOSE] 
     
     Example:
        > python model_utils.py -t /data/dataset/tfmodel/ -k False -l /data/dataset/tflite/ 
                                -s sample_image -verbose True
    """
    parser = argparse.ArgumentParser(description='DESCRIPTION')
    parser.add_argument('-t', '-tfmodel_path', action="store", dest='tfmodel_path', help='DESCRIPTION')
    parser.add_argument('-k', '-keras_model', action="store", dest='keras_model', help='DESCRIPTION')
    parser.add_argument('-l', '-tflite_output', action="store", dest='tflite_output', help='DESCRIPTION')
    parser.add_argument('-s', '-sample_image', action="store", dest='sample_image', help='DESCRIPTION')
    parser.add_argument('-v', '-verbose', action="store", dest='verbose', help='Print log of processing')
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

    main(args)
