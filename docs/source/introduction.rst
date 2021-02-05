Introduction
===========================

.. image:: _static/bioverse.png

The deep-learning module incorporate essential procedures to prepare remote sensing images mostly for automatic classification methods, in this case specifically, using deep-learning approaches. The sections below are organized to explain the main procedures implemented in the Bioverse DL module.

Deep Learning architectures
===========================

Beyond the routines, this code was prepared to have a personal use or even ready for adaptation to be run in server-side. The ain is to incorporate some of the main Deep Learning models for any remote sensing image analysis and mapping. In this version, the following DL architectures were tested:

    - `UNet <https://arxiv.org/abs/1505.04597>`_

UNet
---------------------------
The UNet is convolutional network architecture for fast and precise segmentation of digital images. Until now, the "U shaped" Deep Learning architecture has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks.

UNet, evolved from the traditional convolutional neural network, was first designed and applied in 2015 to process biomedical images, as the paper cited before. As a general convolutional neural network focuses its task on image classification, where input is an image and output is one label, but in biomedical cases, it requires us not only to distinguish whether there is a disease, but also to identify the area.

UNet is dedicated to solving this problem. The reason it is able to identify and delineate the boundaries in a process of a pixel-wise classification. In this sense, the input and output must to be the same size. For example, for an input image of size 2x2, we have the respective :

.. code-block:: python

    [[121, 27], [7, 115]]

where each number is a pixel. Thus, the output will have the same size of 2x2:

.. code-block:: python

    [[0, 1], [0, 1]]

where the outputs could be any number between 0 and 1. The UNet architecture has the following structure:

.. image:: _static/unet.png

The U shape is its characteristic. The architecture is symmetric and consists of two major parts: the left part is called contracting path, which is constituted by the general convolutional process; the right part is expansive path, which is constituted by transposed 2d convolutional layers (you can think it as an upsampling technic for now). Now let’s have a quick look at the implementation:

References
---------------------------

To write this section, the following sources have been used:

    - .. _TowardDataScience: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

Technique
===========================

The detection technique is composed by a set of image processing operators, mainly by Machine Learning techniques, which allow to explore the spectral, spatial and contextual properties of this species in a broad and emerging way. In addition, it is expected that the methodology has reasonable robustness and accuracy, therefore, the methodology is composed of scalable, interoperable, flexible and easily accessible architectures, allowing for any future modifications, experiments or replications.

Image segmentation
---------------------------

In order to classify an image pixel by pixel using a supervised approach, a collection of desired targets is usually needed. This dataset consists of many pairs of image (raster) and its respective label, which have the exact location of the desired objects in the image. These are then presented to a mathematical model that can understand this targets by analysing the possible patterns, spectral similarities, geometries, and many other characteristics that later is easily distinguishable over unknown images. This process is called training, and the module image-processing was developed to prepare the dataset using any satellite image (details of image-processing can be found here). Right after the training, the mathematical model choose can then be used as a predictor.

The prediction or image segmentation procedure involve two types: (i) for images where the dimension is equal to the samples's dimensions used during training, and (ii) images where the dimension is larger. Besides, the inferences have two classes of images, the images without any geographic information, and images with geographic information. The difference is that for images with no geographic metadata, the poligonization (the process to convert PNG prediction in SHP shapefiles - geographic vectors), will not be performed. In this section, we focus specifically how the (ii) was implemented.

Considering a large geographic image as an example, in the figure below is shown how the inference is made. First (a), the large image is tilled in a way that each tile have the same dimension as it was trained.

.. image:: _static/buffer-prediction.png

In order to prevent discontinuous predictions between each tile, a buffer is applied (see (a)). The buffer can be configured also in `settings.py`, with the `BUFFER_TO_INFERENCE` variable, where the integer value represents the number of pixels to apply the buffering. In this way, zero will perform the inferences without buffering. The maximum buffering value is the half of each tile's dimension.

After to predict, each tile will have a correspondent segmentation (see (b)). After to predict every single tile that compose the image, the predictions are then merged (c). Due to the buffering, the discontinuity is minimized during merging. Finally, getting a more consistent map in the end (d).
