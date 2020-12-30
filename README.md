# ECG  Signals Classification with Federated-Learning and Differential Privacy in Keras, Convolutional Neural Network Implementation

This repository contains a more advanced version of the [shallow implementation of ECG classification](https://github.com/mbarzegary/ecg-classification-keras-shallow). It includes federated learning and differential privacy implementation for privacy-preserving machine learning using the [TensorFlow Federated](https://github.com/tensorflow/federated) and [TensorFlow Privacy](https://github.com/tensorflow/privacy) libraries. The code has been used in the following paper, so please cite this if you want to use it in your own research.

    @ARTICLE{Firouzi2020,
    author={F. {Firouzi} and B. {Farahani} and M. {Barzegari} and M. {Daneshmand}},
    journal={IEEE Internet of Things Journal},
    title={AI-Driven Data Monetization: The other Face of Data in IoT-based Smart and Connected Health},
    year={2020},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/JIOT.2020.3027971}}

The code implements a deep neural network for ECG classification using a CNN model. It takes into account the privacy-preserving machine learning considerations, making it a good educational resource/tutorial/show-case for differential privacy and federated learning approaches on complex and deep neural network models. Additionally, the code includes the implementation of some standard image augmentation techniques for preprocessing of the signals.

## Getting started

The code depends on Keras and TensorFlow, and due to the computational-intensive training, it's crucial to have TensorFlow-GPU installed. After installing these components, the code can be executed by running `run_train.py`. The normal, differential privacy, and federated learning training routines are implemented in different functions, so the proper method should be uncommented in `run_train.py`.

You may also have a look at `load_MITBIH.py` to see how the dataset is grouped into various classes for different modes and adjust it. The data can be obtained from the [Kaggle website](https://www.kaggle.com/mondejar/mitbih-database).
