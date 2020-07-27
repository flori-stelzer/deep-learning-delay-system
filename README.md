# deep-learning-delay-system

This repository contains six versions of the deep learning delay system:

* delay-system-classifier-validation: MNIST, Fashion-MNIST, CIFAR-10, SVHN classifications tasks, cross validation on training sets,
* delay-system-classifier-test: MNIST, Fashion-MNIST, CIFAR-10, SVHN classifications tasks, validation on test sets,
* delay-system-denoising-validation: Fashion-MNIST-denoising task, cross validation on training set,
* delay-system-denoising-test: MNIST, Fashion-MNIST-denoising task, validation on test sets,
* delay-system-simple-input-validation: direct input without matrix multiplication, CIFAR-10, SVHN classifications tasks, cross validation on training sets,
* delay-system-simple-input-test:  direct input without matrix multiplication, CIFAR-10, SVHN classifications tasks, validation on test sets.

For details about each version see the readme in the corresponding directory.

In order to run the programs you need to provide the datasets MNIST, Fashion-MNIST, CIFAR-10 and SVHN (cropped version) in folders called "data-MNIST", "data-Fashion-MNIST", "data-CIFAR-10" and "data-SVHN" in a file format readable for the program. Python scripts to parse the original datasets and to create the readable files are provides in the directory "parse-data".
