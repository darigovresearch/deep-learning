import os

from decouple import config

VALID_PREDICTION_EXTENSION = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".tiff", ".TIF", ".TIFF")
DL_DATASET = config('DL_DATASET')

DL_PARAM = {
    'unet': {
        'image_training_folder': os.path.join(DL_DATASET, 'training', 'all'),
        'annotation_training_folder': os.path.join(DL_DATASET, 'training', 'all'),
        'image_validation_folder': os.path.join(DL_DATASET, 'validation', 'all'),
        'annotation_validation_folder': os.path.join(DL_DATASET, 'validation', 'all'),
        'output_prediction': os.path.join(DL_DATASET, 'predictions', '256', 'all', 'inference'),
        'output_checkpoints': os.path.join(DL_DATASET, 'predictions', '256', 'all', 'weight'),
        'save_model_dir': os.path.join(DL_DATASET, 'training', 'all', 'model'),
        'tensorboard_log_dir': os.path.join(DL_DATASET, 'training', 'all', 'log'),
        'pretrained_weights': 'model-78.hdf5',
        'image_prediction_folder': os.path.join(DL_DATASET, 'test'),
        'input_size_w': 256,
        'input_size_h': 256,
        'input_size_c': 3,
        'batch_size': 16,
        'learning_rate': 0.001,
        'filters': 64,
        'kernel_size': 3,
        'deconv_kernel_size': 3,
        'pooling_stride': 2,
        'dropout_rate': 0.5,
        'color_mode': 'rgb',
        'class_mode': None,
        'seed': 1,
        'epochs': 1000,
        'classes': {
                "nut": [102, 153, 0],
                "palm": [153, 255, 153],
                'other': [0, 0, 0]
        },
        'color_classes': {'nut': 1, 'palm': 2, 'other': 0},
        'width_slice': 1000,
        'height_slice': 1000,
    }
}
