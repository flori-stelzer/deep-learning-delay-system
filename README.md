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
* CIFAR-100 dataset
* SVHN (cropped version) dataset

The datasets need to be provided in the corresponding "data-..." folders in a file format readable for the program. Python scripts to parse the original datasets and to create the readable files are provides in the directory "parse-data".

The program does neither require nor use any GPUs and runs on a single thread. 

# Source file details

The core loop is implemented in "main.cpp". Parameters are read in from command line (that process is defined in "katana_get_params.hh"), and most of the variables are initialized in "main.cpp". Depending on the setting, starting weights are either loaded from a file or randomly generated. The weights are adapted using a gradient descent algorithm.

The system can be solved for the delay-case as defined in "solve_dde.cpp", or the equivalent deep neural network can be directly simulated with the functions provided in "solve_network.cpp". 

"global_constants.h" is the file encoding various meta-data about the task at hand, e.g. the length of the sets and the batch sizes. If a different task is desired, the entries in "global_constants.h" must be changed correspondingly. Any change of "global_constants.h" requires a recompilation to take effect.

The local activation function is defined in "f.h" and can be changed there. 

# Example of usage

In the following we describe (as an example of usage) how to perform a cross validation on the MNIST training data with the following options:

* method to choose the delays: random with uniform distribution,
* number of delays: D = 100,
* number of hidden layers: L = 2,
* number of nodes per hidden layer: N = 100,
* node separation: theta = 0.5,
* initial learning rate: eta_0 = 0.01, learning rate scaling factor: eta_1 = 10000,
* noise intensity for regularization of the training images: sigma = 0.1,
* data augmentation option: pixel jittering,
* filename of the text file storing the results: results_MNIST.txt.

Using the program with one of the other datasets (Fashion-MNIST, CIFAR-10, SVHN), for the denoising task, or for valisation on the test set works in an analogous way. One can adjust further options and parameters. A list of all options for each program version is provided in the version-specific readme file in the corresponding directory.

1. Prepare data.
   * Download the files train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz from the MNIST-website: http://yann.lecun.com/exdb/mnist/ and unzip the files.
   * The unzipped files must be stored in a directory with the name "data"; the Python script parse-mnist.py from the folder parse-data must be copied to the directory which contains the "data" directory. Run the Python script to create multiple text files which contain the MNIST data in a format that can be loaded by our program.
   * Move the created text files to the directory "data-MNIST" which must be in the same superdirectory as the folder "delay-system-classifier-validation".
1. Make sure to provide the desired activation function and global constants before compiling the program.
   * The file "f.h" contains the activation function f for the hidden layers, the input preprocessing function g, and the corresponding derivatives f', g'. By default f is the sine function and g is a scaled hyperbolic tangent function. If other functions f and g are desired, they must be changed in "f.h" before compiling the program.
   * The file "global_constants.h" contains 3 sets of constants (one for MNIST/Fashion-MNIST, one for CIFAR-10, and one for SVHN). Make sure that the lines with the correct set of constants are not commented out.
1. Compile the program using the provided makefile, i.e. open a terminal and type "make". (Depending on the configuration of your system, the makefile may need to be modified.) An executable "prog" will be created.
1. Run the program with the desired parameters with the command "./prog -task MNIST -eta0 0.01 -eta1 10000 -input_noise -sigma 0.1 -pixel_shift -diag_method uniform -D 100 -L 2 -N 100 -theta 0.5 -filename results_MNIST.txt".

The program will simulate the delay system and train the parameters for 100 epochs for each of the 6 cross-validation steps. The results will be stored in the text file "results_MNIST.txt".

