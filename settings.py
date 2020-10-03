import os

from decouple import config

DL_DATASET = config('DL_DATASET')

DL_PARAM = {
    'unet': {
        'image_training_folder': os.path.join(DL_DATASET, 'training', 'image'),
        'annotation_training_folder': os.path.join(DL_DATASET, 'training', 'label'),
        'image_validation_folder': os.path.join(DL_DATASET, 'validation', 'image'),
        'annotation_validation_folder': os.path.join(DL_DATASET, 'validation', 'label'),
        'output_prediction': os.path.join(DL_DATASET, 'predictions', '128', 'inference'),
        'output_checkpoints': os.path.join(DL_DATASET, 'predictions', '128', 'weight'),
        'pretrained_weights': '',
        'input_size_w': 128,
        'input_size_h': 128,
        'input_size_c': 3,
        'batch_size': 16,
        'filters': 64,
        'color_mode': 'grayscale',
        'seed': 1,
        'epochs': 3000,
        'classes': {
                "nut": [102, 153, 0],
                "palm": [153, 255, 153],
                'other': [0, 0, 0]
        }
    },
    'deeplabv3': {
        'image_training_folder': os.path.join(DL_DATASET, 'training', 'image'),
        'annotation_training_folder': os.path.join(DL_DATASET, 'training', 'label'),
        'image_validation_folder': os.path.join(DL_DATASET, 'validation', 'image'),
        'annotation_validation_folder': os.path.join(DL_DATASET, 'validation', 'label'),
        'output_prediction': os.path.join(DL_DATASET, 'predictions', '128', 'inference'),
        'output_checkpoints': os.path.join(DL_DATASET, 'predictions', '128', 'weight'),
        'pretrained_weights': '',
        'input_size_w': 128,
        'input_size_h': 128,
        'input_size_c': 3,
        'batch_size': 16,
        'filters': 64,
        'color_mode': 'grayscale',
        'seed': 1,
        'epochs': 3000,
        'classes': {
                "nut": [102, 153, 0],
                "palm": [153, 255, 153],
                'other': [0, 0, 0]
        }
    },
}
