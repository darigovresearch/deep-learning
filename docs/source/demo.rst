******************
Demo and Examples
******************

Examples
===================

Training the model
-------------------

.. code-block:: bash

    python main.py -model unet -train True -predict False -verbose True

the runtime logging will print something like (note that all CUDA libraries must to be all loaded correctly, otherwise, it will not be registered and then CPU is used):

.. code-block:: bash

    (.venv) user@user-machine:~/deep-learning$ python main.py -model unet -train True -predict False -verbose True
    2020-09-23 20:43:31.621367: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    [2020-09-23 20:43:42] {main.py        :63  } INFO : Starting process...
    [2020-09-23 20:43:42] {main.py        :39  } INFO : >> UNET model selected...
    [2020-09-23 20:43:42] {unet.py        :19  } INFO : >>>> Settings up UNET model...
    2020-09-23 20:43:43.281992: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
    2020-09-23 20:43:43.654774: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-09-23 20:43:43.655488: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties:
    pciBusID: 0000:00:04.0 name: Tesla P4 computeCapability: 6.1
    coreClock: 1.1135GHz coreCount: 20 deviceMemorySize: 7.43GiB deviceMemoryBandwidth: 178.99GiB/s
    2020-09-23 20:43:43.655531: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-09-23 20:43:43.655713: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
    2020-09-23 20:43:44.191179: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-09-23 20:43:44.398087: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-09-23 20:43:45.234487: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-09-23 20:43:45.458019: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-09-23 20:43:46.809363: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-09-23 20:43:46.809421: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1753] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...
    2020-09-23 20:43:46.809732: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2020-09-23 20:43:47.219185: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2300000000 Hz
    2020-09-23 20:43:47.220090: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x27b8e30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-09-23 20:43:47.220121: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-09-23 20:43:47.237334: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-09-23 20:43:47.237365: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]
    [2020-09-23 20:43:48] {unet.py        :74  } INFO : >>>> Done!
    Found 2636 images belonging to 1 classes.
    Found 2636 images belonging to 1 classes.
    Epoch 1/500
     4/50 [=>............................] - ETA: 2:40 - loss: 0.6931 - accuracy: 0.8943
    ...

Predicting with an existent weight
----------------------------------

