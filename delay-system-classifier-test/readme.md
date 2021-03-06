# Simulation Program for the Deep Learning Delay System

The following adaptations have to be done before compiling the code:

1. If you want to use a different activation function than f = sin and/or a different input preprocessing function than g = tanh, you need to modify the corresponding functions in f.h.

2. If you want to use a task other than MNIST or Fashion-MNIST, you need to uncomment the corresponding lines in the file global_constants.h.

The program can be excecuted with the following option:

| flag                                   | default value     | explanation                                                                                                                                        |
| -------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| -filename [filename.txt]               | filename.txt      | Sets the filename of the textfile containing the simulation results.                                                                               |
| -task [name]                           | MNIST             | MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100-coarse and SVHN are available options.                                                                                     |
| -system_simu [name]                    | dde_ibp           | Choose dde_ibp for simulation with delay system, choose decoupled_network for simulation with MLP.                                                 |
| -grad_comp [name]                      | backprop_standard | Choose backprop_standard for dde_ibp and backprop_classic for decoupled_network.                                                                   |
| -gradient_check                        | false             | Computes cosine similarity between backprop gradient and numerically computed gradient. Only availble for "dde_ibp"                                |
| -N [integer number]                    | 100               | Number of nodes per hidden layer.                                                                                                                  |
| -L [integer number]                    | 2                 | Number of hidden layers.                                                                                                                           |
| -D [integer number]                    | 50                | Number of delays resp. diagonals in the hidden weight matrix.                                                                                      |
| -theta [decimal number]                | 0.5               | Node separation.                                                                                                                                   |
| -diag_method [name of method]          | uniform           | uniform, equidist or from_file                                                                                                                     |
| -diag_distance [integer number]        | 0                 | Distance between diagonals if diag_method is equi_dist, minimum distance between diagonals if diag_method is uniform, otherwise ignored.           |
| -diag_file_path [file path]            | diag.txt          | File path if diag_method is equi_dist.                                                                                                             |
| -make_diags                            | false             | The program will create a textfile diag.txt with diagonal indices generated by the given method which can be use by from_file.                     |
| -number_of_epochs [integer number]     | 100               | Number of epochs.                                                                                                                                  |
| -eta0 [decimal number]                 | 0.001             | learning rate eta = min(eta_0, eta_1 / step)                                                                                                       |
| -eta1 [decimal number]                 | 1000.0            |                                                                                                                                                    |
| -pixel_shift                           | false             | Enable random pixel shift for data augmentation.                                                                                                   |
| -max_pixel_shift [integer number]      | 1                 | Maximum distance for pixel shift.                                                                                                                  |
| -training_noise                        | false             | Enable noise for data augmentation.                                                                                                                |
| -max_pixel_shift [integer number]      | 0.01              | Standard deviation of Gaussian training noise.                                                                                                     |
| -rotation                              | false             | Enable random rotation for data augmentation. Use only for CIFAR-10.                                                                               |
| -max_rotation_degrees [decimal number] | 15.0              | Maximum degree for rotation.                                                                                                                       |
| -flip                                  | false             | Enable random horizontal flip for data augmentation. Use only for CIFAR-10.                                                                        |
| -dropout_rate [decimal number]         | 0.0               | Dropout rate for training.                                                                                                                         |
