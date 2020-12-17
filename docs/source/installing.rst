*************
Installation
*************

Configuration of Project Environment
====================================

Setting Python and virtual environment (Linux distribution)
-----------------------------------------------------------

First of all, check if you have installed the libraries needed:

.. code-block:: bash

  sudo apt-get install python3-env

then, in the

.. code-block:: bash

  python -m venv .venv

and activates it:

.. code-block:: bash

  source .venv/bin/activate

as soon you have it done, you are ready to install the requirements.

Preparing your `.env` file
--------------------------

This library uses decoupling, which demands you to set up variables that is only presented locally, for instance, the path you want to save something, or the resources of your project. In summary, your environment variables. So, copy a paste the file ``.env-example`` and rename it to ``.env``. Afterwards, just fill out each of the variables content within the file:

.. code-block:: python

   DL_DATASET=PATH_TO_TRAINING_FOLDER


Installing `requirements.txt`
------------------------------

If you do not intent to use GPU, there is no need to install support to it. So, in requirements file, make sure to set ``tensorflow-gpu`` to only ``tensorflow``. If everything is correct, and you **virtualenv** is activated, execute:

.. code-block:: bash

   pip install -r requirements.txt


The `settings.py` file
--------------------------

This file centralized all constants variable used in the code, in particular, the constants that handle all the DL model. Thus, the Python dictionary ``DL_PARAM`` splits all the values and parameters by model type. In this case, only the UNet architecture was implemented:

.. code-block:: python

   import os

   from decouple import config

   DL_DATASET = config('DL_DATASET')

   DL_PARAM = {
       'unet': {
            PARAMS OF UNET
            },
       'deeplabv3': {
            PARAMS OF DEEPLABV3+
            }
       ...
   }

in this way, if a new model is introduced to the code, a new key is add to this dictionary with its respective name, then, it will automatically load all the parameters according to the type of mode the user choose in the ``-model`` command-line option. For instance, if a `PSPNet
<https://arxiv.org/abs/1612.01105v2>`_ is included:

.. code-block:: python

   DL_PARAM = {
       'unet': {
           'image_training_folder': os.path.join(DL_DATASET, 'samples', LABEL_TYPE, 'training'),
           'annotation_training_folder': os.path.join(DL_DATASET, 'samples', LABEL_TYPE, 'training'),
           'image_validation_folder': os.path.join(DL_DATASET, 'samples', LABEL_TYPE, 'validation'),
           'annotation_validation_folder': os.path.join(DL_DATASET, 'samples', LABEL_TYPE, 'validation'),
           'output_prediction': os.path.join(DL_DATASET, 'predictions', '256', 'all', 'inference', 'png'),
           'output_prediction_shp': os.path.join(DL_DATASET, 'predictions', '256', 'inference', 'shp'),
           'output_checkpoints': os.path.join(DL_DATASET, 'predictions', '256', 'weight'),
           'output_history': os.path.join(DL_DATASET, 'predictions', '256', 'history'),
           'save_model_dir': os.path.join(DL_DATASET, 'samples', LABEL_TYPE, 'training', 'model'),
           'tensorboard_log_dir': os.path.join(DL_DATASET, 'samples', LABEL_TYPE, 'training', 'log'),
           'image_prediction_folder': os.path.join(DL_DATASET, 'test'),
           'image_prediction_tmp_slice_folder': os.path.join(DL_DATASET, 'tmp_slice'),
           'pretrained_weights': 'model-input256-256-batch16-drop05-epoch98.hdf5',
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
           'epochs': 100,
           'classes': {
                   "other": [0, 0, 0],
                   "nut": [102, 153, 0],
                   "palm": [153, 255, 153]
           },
           'color_classes': {0: [0, 0, 0], 1: [102, 153, 0], 2: [153, 255, 153]},
           'width_slice': 1000,
           'height_slice': 1000,
       }
   }

The hierarchy of folders
--------------------------

It is very recommended to prepare the hierarchy of folders as described in this section. When the training samples are build (as described in `bioverse image-processing module
<https://github.com/Bioverse-Labs/image-processing>`_), four main folders are created: one for raster, one for the annotation (i.e. ground-truth, label, reference images), one to save the predictions (i.e. inferences), and finally one to store the validation samples. Besides, in order to conduct multiple test, such as different dimensions and classes of training samples, subfolders are also created under each folder, such as:

::

   samples
   │   └── classid
   │      ├── training
   │      │   ├── image
   |      │   |    :: images in TIF extension
   │      │   ├── label
   |      │   |    :: annotation in PNG extension
   │      │   ├── log
   │      │   └── model
   │      └── validation
   │          ├── image
   |            :: images in TIF extension
   │          └── label
   |            :: annotation in PNG extension
   ├── predictions
   │   └── 256
   │       ├── history
   │       ├── inference
   │       └── weight


This suggestion of folder hierarchy is not mandatory, just make sure the paths is correctly pointed in ``settings.py`` file.

NVIDIA's driver and CUDA for Ubuntu 20.4
========================================

For most of the processing and research approaching Deep Learning (DL) methodologies, a certain computational power is needed. Recently, the use of GPUs has expanded the horizon of heavy machine learning processing such as the DL demands.

TensorFlow GPU support requires an assortment of drivers and libraries. To simplify installation and avoid library conflicts, the TensorFlow recommends the use of a `TensorFlow Docker Image
<https://www.tensorflow.org/install/docker>`_, which incorporates all the setups needed to this kind of procedure. For more details, please, access the `TensorFlow official page
<https://www.tensorflow.org/install/gpu>`_.

Considering not using a Docker image, there are many tutorials in how to install your NVIDIA's driver and CUDA toolkit, such as described in `Google Cloud
<https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#ubuntu-driver-steps>`_, `NVIDIA Dev
<https://developer.nvidia.com/cuda-downloads>`_, or even a particular pages. The way presented here, could vary in your need. So, if you want to prepare your environment based in a complete different OS or architecture, just follow the right steps in the provider website, and make sure to have all the cuda libraries listed in ``LD_LIBRARY_PATH``.

So, for Linux OS, x86_64 arch, Tensorflow 2.1+, and Ubuntu LTS 20.04, first, it is necessary to install all software requirements, which includes:

NVIDIA® GPU drivers — CUDA® 10.1 requires 418.x or higher.
CUDA® Toolkit — TensorFlow supports CUDA® 10.1 (TensorFlow >= 2.1.0)
CUPTI ships with the CUDA® Toolkit.
cuDNN SDK 7.6
(Optional) TensorRT 6.0 to improve latency and throughput for inference on some models.
to install CUDA® 10 (TensorFlow >= 1.13.0) on Ubuntu 16.04 and 18.04. These instructions may work for other Debian-based distros.

.. code-block:: bash

   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
   sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
   sudo apt-get update
   sudo apt-get -y install cuda

then, reboot the system.

As mentioned before, TensorFlow will seek for some of the CUDA libraries during training. As reported by many users, is possible that some of them is installed in different location in your filesystem. To guarantee your ``LD_LIBRARY_PATH`` is pointing out to the right folder, add the

If you followed all steps and have it installed properly, then, the final steps is So, add the following lines to your ``~\.bashrc`` file using nano or any other editor (check the version to replace on XXX):

.. code-block:: bash

   export PATH=/usr/local/cuda-XXX/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-XXX/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

right after, updated it:

.. code-block:: bash

   source ~/.bashrc

If you followed all steps and have it installed properly, you are ready to train your model!

For more details, follow the issue reported `here
<https://askubuntu.com/questions/1145946/tensorflow-wont-import-with-sudo-python3>`_  and `here
<https://stackoverflow.com/questions/60208936/cannot-dlopen-some-gpu-libraries-skipping-registering-gpu-devices>`_.
