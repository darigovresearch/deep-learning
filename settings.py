from decouple import config

DL_DATASET = config('DL_DATASET', default='/data/prodes/dl/cloud/dataset/')

DL_PARAM = {
    'unet': {
        'image_training_folder': DL_DATASET + 'ruemonge2014/dataset/unet/image/train/',
        'annotation_training_folder': DL_DATASET + 'ruemonge2014/dataset/unet/annotation/',
        'image_prediction_folder': DL_DATASET + 'ruemonge2014/dataset/unet/image/test/',
        'output_prediction': DL_DATASET + 'ruemonge2014/output/unet/',
        'output_checkpoints': DL_DATASET + 'ruemonge2014/weight/unet/',
        # 'pretrained_weights': DL_DATASET + 'ruemonge2014/output/ruemonge2014/weight/unet/.34',
        'pretrained_weights': '',
        'input_size_w': 768,
        'input_size_h': 960,
        'input_size_c': 3,
        'batch_size': 2,
        'filters': 64,
        'color_mode': 'grayscale',
        'seed': 1,
        'epochs': 150,
        'classes': {
                'sky': [0, 255, 255],
                'roof': [0, 0, 255],
                'wall': [255, 255, 0],
                'window': [255, 0, 0],
                'door': [255, 128, 0],
                'store': [0, 255, 0],
                'balcony': [255, 0, 255],
                'other': [0, 0, 0],
        }
    },
    'pspnet': {
        'image_training_folder': DL_DATASET + 'ruemonge2014/dataset/pspnet/image/train/',
        'annotation_training_folder': DL_DATASET + 'ruemonge2014/dataset/pspnet/annotation/output/',
        'image_prediction_folder': DL_DATASET + 'ruemonge2014/dataset/pspnet/image/test/',
        'output_prediction': DL_DATASET + 'ruemonge2014/output/pspnet/',
        'output_checkpoints': DL_DATASET + 'ruemonge2014/weight/pspnet/',
        'pretrained_weights': DL_DATASET + 'ruemonge2014/output/ruemonge2014/weight/pspnet/.14',
        'input_size_w': 768,
        'input_size_h': 960,
        'input_size_c': 3,
        'batch_size': 16,
        'epochs': 150,
        'classes': {
                'sky': [0, 255, 255],
                'roof': [0, 0, 255],
                'wall': [255, 255, 0],
                'window': [255, 0, 0],
                'door': [255, 128, 0],
                'store': [0, 255, 0],
                'balcony': [255, 0, 255],
                'other': [0, 0, 0],
        }
    },
    'segnet': {
        'image_training_folder': DL_DATASET + 'ruemonge2014/dataset/segnet/image/train/',
        'annotation_training_folder': DL_DATASET + 'ruemonge2014/dataset/segnet/annotation/',
        'image_prediction_folder': DL_DATASET + 'ruemonge2014/dataset/segnet/image/test/',
        'output_prediction': DL_DATASET + 'ruemonge2014/output/segnet/',
        'output_checkpoints': DL_DATASET + 'ruemonge2014/weight/segnet/',
        'pretrained_weights': DL_DATASET + 'ruemonge2014/output/ruemonge2014/weight/segnet/.14',
        'input_size_w': 768,
        'input_size_h': 960,
        'input_size_c': 3,
        'batch_size': 16,
        'epochs': 150,
        'classes': {
                'sky': [0, 255, 255],
                'roof': [0, 0, 255],
                'wall': [255, 255, 0],
                'window': [255, 0, 0],
                'door': [255, 128, 0],
                'store': [0, 255, 0],
                'balcony': [255, 0, 255],
                'other': [0, 0, 0],
        }
    }

}
