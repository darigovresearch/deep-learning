import os

from decouple import config

DL_DATASET = config('DL_DATASET')
GEOGRAPHIC_ACCEPT_EXTENSION = ('.TIFF', '.tiff', '.TIF', '.tif', '.GEOTIFF', '.geotiff')
NON_GEOGRAPHIC_ACCEPT_EXTENSION = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

LABEL_TYPE = 'classid'
VALIDATION_SPLIT = 0.15
RASTER_TILES_COMPOSITION = ['5', '3', '2']
CLASSES_TO_CONVERT_RASTER_TO_GEOGRAPHIC_FORMAT = ['nut']
BUFFER_TO_INFERENCE = 30


# TODO: remove paths from network dict
DL_PARAM = {
    'unet': {
        'image_training_folder': os.path.join(DL_DATASET, 'samples', LABEL_TYPE),
        'annotation_training_folder': os.path.join(DL_DATASET, 'samples', LABEL_TYPE),
        'output_checkpoints': os.path.join(DL_DATASET, 'predictions', '256', 'weight'),
        'output_history': os.path.join(DL_DATASET, 'predictions', '256', 'history'),
        'save_model_dir': os.path.join(DL_DATASET, 'predictions', '256', 'model'),
        'tensorboard_log_dir': os.path.join(DL_DATASET, 'predictions', '256', 'log'),
        'pretrained_weights': 'model-input256-256-batch16-drop05-epoch67.hdf5',
        'image_prediction_folder': os.path.join(DL_DATASET, 'test', 'big'),
        'output_prediction': os.path.join(DL_DATASET, 'predictions', '256', 'inference'),
        'output_prediction_shp': os.path.join(DL_DATASET, 'predictions', '256', 'shp'),
        'tmp_slices': os.path.join(DL_DATASET, 'tmp', 'tmp_slice'),
        'tmp_slices_predictions': os.path.join(DL_DATASET, 'tmp', 'tmp_slice_predictions'),
        'input_size_w': 256,
        'input_size_h': 256,
        'input_size_c': 3,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'filters': 32,
        'kernel_size': 3,
        'deconv_kernel_size': 3,
        'pooling_size': 2,
        'pooling_stride': 2,
        'dropout_rate': 0.5,
        'epochs': 500,
        'classes': {
                "other": [0, 0, 0],
                "nut": [102, 153, 0],
                "palm": [153, 255, 153]
        },
        'color_classes': {0: [0, 0, 0], 1: [102, 153, 0], 2: [153, 255, 153]},
        'width_slice': 256,
        'height_slice': 256
    }
}

