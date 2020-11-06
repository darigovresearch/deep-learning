import os

from decouple import config

VALID_ENTRIES_EXTENSION = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".tiff", ".TIF", ".TIFF")
VALID_PREDICTION_EXTENSION = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".tiff", ".TIF", ".TIFF")
DL_DATASET = config('DL_DATASET')

DL_PARAM = {
    'unet': {
        'image_training_folder': os.path.join(DL_DATASET, 'training', 'all'),
        'annotation_training_folder': os.path.join(DL_DATASET, 'training', 'all'),
        'image_validation_folder': os.path.join(DL_DATASET, 'validation', 'all'),
        'annotation_validation_folder': os.path.join(DL_DATASET, 'validation', 'all'),
        'output_prediction': os.path.join(DL_DATASET, 'predictions', '256', 'all', 'inference', 'png'),
        'output_prediction_shp': os.path.join(DL_DATASET, 'predictions', '256', 'all', 'inference', 'shp'),
        'output_checkpoints': os.path.join(DL_DATASET, 'predictions', '256', 'all', 'weight'),
        'output_history': os.path.join(DL_DATASET, 'predictions', '256', 'all', 'history'),
        'save_model_dir': os.path.join(DL_DATASET, 'training', 'all', 'model'),
        'tensorboard_log_dir': os.path.join(DL_DATASET, 'training', 'all', 'log'),
        'image_prediction_folder': os.path.join(DL_DATASET, 'test'),
        'image_prediction_tmp_slice_folder': os.path.join(DL_DATASET, 'tmp_slice'),
        'pretrained_weights': 'model-input256-256-batch16-drop05-epoch98.hdf5',
        'input_size_w': 256,
        'input_size_h': 256,
        'input_size_c': 3,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'filters': 64,
        'kernel_size': 3,
        'deconv_kernel_size': 3,
        'pooling_stride': 2,
        'dropout_rate': 0.5,
        'color_mode': 'rgb',
        'class_mode': 'categorical',
        'seed': 1,
        'epochs': 100,
        'classes': {
                "nut": [102, 153, 0],
                "palm": [153, 255, 153],
                'other': [0, 0, 0]
        },
        'color_classes': {0: [102, 153, 0], 1: [153, 255, 153], 2: [0, 0, 0]},
        'width_slice': 1000,
        'height_slice': 1000,
    }
}

