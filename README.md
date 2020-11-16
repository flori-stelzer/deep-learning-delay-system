# deep-learning-delay-system

This github provides the code used for implementing a deep neural network with a single neuron. 

# Overview

This repository contains four versions of the deep learning delay system, each individually compiled for a specific set of tasks. We have also separated the test and training sets:

* delay-system-classifier-validation: MNIST, Fashion-MNIST, CIFAR-10, SVHN classifications tasks, cross validation on training sets,
* delay-system-classifier-test: MNIST, Fashion-MNIST, CIFAR-10, SVHN classifications tasks, validation on test sets,
* delay-system-denoising-validation: Fashion-MNIST-denoising task, cross validation on training set,
* delay-system-denoising-test: MNIST, Fashion-MNIST-denoising task, validation on test sets,

For details about each version see the readme in the corresponding directory. 

# Setup

The program is written in C++ and requires at least the C++11 standard. Makefiles are provided. 

In addition to the code and standard C++ packages, the following are required:

* The armadillo linear algebra package http://arma.sourceforge.net/
* MNIST dataset
* Fashion-MNIST dataset
* CIFAR-10 dataset
* SVHN (cropped version) dataset

The datasets need to be provided in the corresponding "data-..." folders in a file format readable for the program. Python scripts to parse the original datasets and to create the readable files are provides in the directory "parse-data".

The program does neither require nor use any GPUs and runs on a single thread. 

# Source file details

The core loop is implemented in "main.cpp". Parameters are read in from command line (that process is defined in "katana_get_params.hh"), and most of the variables are initialized in "main.cpp". Depending on the setting, starting weights are either loaded from a file or randomly generated. The weights are adapted using a gradient descent algorithm.

The system can be solved for the delay-case as defined in "solve_dde.cpp", or the equivalent deep neural network can be directly simulated with the functions provided in "solve_network.cpp". 

"global_constants.h" is the file encoding various meta-data about the task at hand, e.g. the length of the sets and the batch sizes. If a different task is desired, the entries in "global_constants.h" must be changed correspondingly. Any change of "global_constants.h" requires a recompilation to take effect.  

The local activation function is defined in "f.h" and can be changed there. 
