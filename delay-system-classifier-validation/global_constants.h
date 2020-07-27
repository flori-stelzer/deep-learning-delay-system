#ifndef GLOBAL_CONSTANTS_H
#define GLOBAL_CONSTANTS_H

namespace globalconstants
{
	// uncomment if task is MNIST or Fashion-MNIST:
    const int M = 784;  // number of features = number of input nodes
	const int P = 10;  // number of classes = number of output nodes
	const int number_of_training_batches = 6;
    const int training_batch_size = 10000;
    const int test_batch_size = 10000;
    
    // uncomment if task is CIFAR-10:
    //const int M = 3072;  // number of features = number of input nodes
	//const int P = 10;  // number of classes = number of output nodes
	//const int number_of_training_batches = 5;
    //const int training_batch_size = 10000;
    //const int test_batch_size = 10000;
    
    // uncomment if task is CIFAR-2:
    //const int M = 3072;  // number of features = number of input nodes
	//const int P = 2;  // number of classes = number of output nodes
	//const int number_of_training_batches = 5;
    //const int training_batch_size = 1939;
    //const int test_batch_size = 2000;
    
    // uncomment if task is SVHN:
    //const int M = 3072;  // number of features = number of input nodes
	//const int P = 10;  // number of classes = number of output nodes
	//const int number_of_training_batches = 6;
    //const int training_batch_size = 12209;
    //const int test_batch_size = 26032;
	
}

#endif
