import settings

from keras.preprocessing.image import ImageDataGenerator


class DL:
    def __init__(self):
        pass

    def training_generator(self, network_type, is_augment):
        """
        The input generator. Load the images and augment them (if is_augment is set True).

        Source:
            - https://github.com/zhixuhao/unet/blob/master/data.py
            - hierarchy of folders: https://stackoverflow.com/questions/58050113/imagedatagenerator-for-semantic-segmentation
            - color_mode: https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class

        ** Note:
            This procedure seems not to work with multiclass image segmentation. The images is all empty,
            there is a huge] misunderstanding about labeling type required for this task (i.e. should the label be
            built with class id, onehot-encoded, or rgb??). Nothing really clear in Keras documentation.

        :param network_type: the deep learning architecture's name
        :param is_augment: if True, the augmentation transformations are applied
        :return train_generator: the tuples list with raster and respective label
        """
        if is_augment is True:
            data_gen_args = dict(rescale=1. / 255, rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
        else:
            data_gen_args = dict(rescale=1. / 255)

        load_param = settings.DL_PARAM[network_type]
        
        train_datagen = ImageDataGenerator(**data_gen_args)
        val_datagen = ImageDataGenerator(**data_gen_args)

        train_image_generator = train_datagen.flow_from_directory(
            load_param['image_training_folder'],
            classes=['image'],
            class_mode=load_param['class_mode'],
            target_size=(load_param['input_size_w'],
                         load_param['input_size_h']),
            seed=load_param['seed'],
            color_mode=load_param['color_mode'],
            batch_size=load_param['batch_size'],
            shuffle=True)
        train_label_generator = train_datagen.flow_from_directory(
            load_param['annotation_training_folder'],
            classes=['label'],
            class_mode=load_param['class_mode'],
            target_size=(load_param['input_size_w'],
                         load_param['input_size_h']),
            seed=load_param['seed'],
            color_mode='grayscale',
            batch_size=load_param['batch_size'],
            shuffle=True)

        val_image_generator = val_datagen.flow_from_directory(
            load_param['image_validation_folder'],
            classes=['image'],
            class_mode=load_param['class_mode'],
            target_size=(load_param['input_size_w'],
                         load_param['input_size_h']),
            seed=load_param['seed'],
            color_mode=load_param['color_mode'],
            batch_size=load_param['batch_size'],
            shuffle=True)
        val_label_generator = val_datagen.flow_from_directory(
            load_param['annotation_validation_folder'],
            classes=['label'],
            class_mode=load_param['class_mode'],
            target_size=(load_param['input_size_w'],
                         load_param['input_size_h']),
            seed=load_param['seed'],
            color_mode='grayscale',
            batch_size=load_param['batch_size'],
            shuffle=True)

        train_generator = zip(train_image_generator, train_label_generator)
        val_generator = zip(val_image_generator, val_label_generator)

        return train_generator, val_generator
