import os

from decouple import config

DL_DATASET = config('DL_DATASET')
GEOGRAPHIC_ACCEPT_EXTENSION = ('.TIFF', '.tiff', '.TIF', '.tif', '.GEOTIFF', '.geotiff')
NON_GEOGRAPHIC_ACCEPT_EXTENSION = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

LABEL_TYPE = 'classid'
PLOT_TRAINING = True
VALIDATION_SPLIT = 0.15
RASTER_TILES_COMPOSITION = ['1', '2', '3']
CLASSES_TO_CONVERT_RASTER_TO_GEOGRAPHIC_FORMAT = ['acai']
BUFFER_TO_INFERENCE = 30
INDEX_OF_BACKGROUND_COLOR = 0

# DL_PARAM = {
#     'unet': {
#         'image_training_folder': os.path.join(DL_DATASET, 'samples', LABEL_TYPE),
#         'annotation_training_folder': os.path.join(DL_DATASET, 'samples', LABEL_TYPE),
#         'output_checkpoints': os.path.join(DL_DATASET, 'predictions', 'weight'),
#         'output_history': os.path.join(DL_DATASET, 'predictions', 'history'),
#         'save_model_dir': os.path.join(DL_DATASET, 'predictions', 'model'),
#         'tensorboard_log_dir': os.path.join(DL_DATASET, 'predictions', 'log'),
#         'pretrained_weights': 'model-input256-256-drop05-epoch24.hdf5',
#         'image_prediction_folder': os.path.join(DL_DATASET, 'test'),
#         'output_prediction': os.path.join(DL_DATASET, 'predictions', 'inference', 'png'),
#         'output_prediction_shp': os.path.join(DL_DATASET, 'predictions', 'inference', 'shp'),
#         'tmp_slices': os.path.join(DL_DATASET, 'tmp', 'tmp_slice'),
#         'tmp_slices_predictions': os.path.join(DL_DATASET, 'tmp', 'tmp_slice_predictions'),
#         'input_size_w': 256,
#         'input_size_h': 256,
#         'input_size_c': 3,
#         'batch_size': 16,
#         'learning_rate': 0.0001,
#         'filters': 32,
#         'kernel_size': 3,
#         'deconv_kernel_size': 3,
#         'pooling_size': 2,
#         'pooling_stride': 2,
#         'dropout_rate': 0.5,
#         'epochs': 300,
#         'classes': {
#                 "other": [0, 0, 0],
#                 "nut": [102, 153, 0],
#                 "palm": [153, 255, 153]
#         },
#         'color_classes': {0: [0, 0, 0], 1: [102, 153, 0], 2: [153, 255, 153]},
#         'width_slice': 256,
#         'height_slice': 256
#     }
# }

DL_PARAM = {
    'unet': {
        'image_training_folder': os.path.join(DL_DATASET, 'samples'),
        'annotation_training_folder': os.path.join(DL_DATASET, 'samples'),
        'output_checkpoints': os.path.join(DL_DATASET, 'predictions', 'weight'),
        'output_history': os.path.join(DL_DATASET, 'predictions', 'history'),
        'save_model_dir': os.path.join(DL_DATASET, 'predictions', 'model'),
        'tensorboard_log_dir': os.path.join(DL_DATASET, 'predictions', 'log'),
        'pretrained_weights': 'model-input256-256-drop05-epoch41.hdf5',
        'image_prediction_folder': os.path.join(DL_DATASET, 'test'),
        'output_prediction': os.path.join(DL_DATASET, 'predictions', 'inference', 'png'),
        'output_prediction_shp': os.path.join(DL_DATASET, 'predictions', 'inference', 'shp'),
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
        'epochs': 150,
        'classes': {
                "other": [0, 0, 0],
                "acai": [102, 153, 0]
        },
        'color_classes': {0: [0, 0, 0], 1: [102, 153, 0]},
        'width_slice': 256,
        'height_slice': 256
    }
}

