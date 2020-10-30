#include "solve_network.h"

using namespace std;
using namespace arma;
using namespace globalconstants;


void solve_network(mat &activations, mat &node_states, double (&output_activations)[P], double (&outputs)[P], vec &g_primes,
					vec &input_data, mat input_weights, cube hidden_weights, mat output_weights, double theta, double alpha,
					int N, int L)
{
	/*
	Function to solve the network equations.
	Provides states of the hidden and output nodes and their activations.
	
	Args:
	activations:        reference to arma::mat with L rows, N cols.
	                    To be filled with activations of the hidden nodes.
				        Each row represents a layer.
	node_states:        reference to arma::mat with L rows, N cols.
	                    To be filled with states of the hidden nodes.
	output_activations: reference to double array of size P = 10.
	                    To be filled with the activations of the output nodes.
	outputs:            reference to double array of size P = 10.
	                    To be filled with the output node states.
	input_data:         reference to arma::vec of length M.
	                    Input vector. Contains the pixel values of an input image.
	input_weights:      reference to arma::mat of size N x (M + 1)
	                    Matrix W^in. Contains the weights connecting the input layer
					    to the first hidden layer (including the input bias weight).
	hidden_weights:     reference to arma::cube with L - 1 rows, N cols, N + 1 slices.
	                    Contains the weights between the hidden layers.
						1st index: layer (where the connection ends),
						2nd index: node index of layer where the connection ends,
						3rd index: node index of layer where the connections begins.
	output_weights:     reference to arma::mat with size P x (N + 1) (where P = 10).
	                    Contains the weights connecting the last hidden layer to the output layer.
	theta:              double.
	                    Node Separation. theta = T / N.
	alpha:              double.
	                    Factor for the linear dynamics. Must be negative.
	N:                  int.
						Nodes per hidden layer (except bias node).
	L:                  int.
						Number of hidden layers.
	*/
	
	double x_0 = 0.0;
	
	double exp_factor = exp(alpha * theta);
	double phi = (exp_factor - 1.0) / alpha;
	
	double summe = 0.0;
	
	
	// We compute the activations of the first layer:
	for (int n = 0; n < N; ++n){
		summe = input_weights(n, M);  // bias weight
		for (int m = 0; m < M; ++m){
			summe += input_weights(n, m) * input_data(m);
		}
		g_primes(n) = input_processing_prime(summe);
		activations(0, n) = input_processing(summe);
	}	
	
	// compute node states for first hidden layer
	node_states(0, 0) = exp_factor * x_0 + phi * f(activations(0, 0));
	for (int n = 1; n < N; ++n){
		node_states(0, n) = exp_factor * node_states(0, n - 1) + phi * f(activations(0, n));
	}
	
	// compute activations and node states for hidden layers 2, ..., L.
	// NOTE: hidden_weights contains weights for layers 2, ..., L,
	//       i.e. hidden_weights(0, n, j) is input to 2nd hidden layer,
	//            hidden_weights(1, n, j) is input to 3rd hidden layer etc.
	for (int l = 1; l < L; ++l){
		// compute activations
		for (int n = 0; n < N; ++n){
			summe = hidden_weights(l - 1, n, N);  // bias weight
			for (int j = 0; j < N; ++j){
				summe += hidden_weights(l - 1, n, j) * node_states(l - 1, j);
			}
			activations(l, n) = summe;
		}
		// compute node states
		node_states(l, 0) = exp_factor * node_states(l - 1, N - 1) + phi * f(activations(l, 0));
		for (int n = 1; n < N; ++n){
			node_states(l, n) = exp_factor * node_states(l, n - 1) + phi * f(activations(l, n));
		}
	}
	
	// compute output activations
	for (int p = 0; p < P; ++p){
		summe = output_weights(p, N);  // bias weight
		for (int n = 0; n < N; ++n){
			summe += output_weights(p, n) * node_states(L - 1, n);
		}
		output_activations[p] = summe;
	}
	
	// compute outputs with clipping activation function
	for (int p = 0; p < P; ++p){
		if (output_activations[p] < 0.0){
			outputs[p] = 0.0;
		} else if (output_activations[p] > 1.0){
			outputs[p] = 1.0;
		} else {
			outputs[p] = output_activations[p];
		}
	}
}


void solve_network_decoupled(mat &activations, mat &node_states, double (&output_activations)[P], double (&outputs)[P], vec &g_primes,
					vec &input_data, mat input_weights, cube hidden_weights, mat output_weights, int N, int L)
{
	/*
	Function to solve the decoupled network equations, i.e. a classical multi-layer neural network.
	This approximates the delay network for large theta.
	Provides states of the hidden and output nodes and their activations.
	
	Args:
	activations:        reference to arma::mat with L rows, N cols.
	                    To be filled with activations of the hidden nodes.
				        Each row represents a layer.
	node_states:        reference to arma::mat with L rows, N cols.
	                    To be filled with states of the hidden nodes.
	output_activations: reference to double array of size P = 10.
	                    To be filled with the activations of the output nodes.
	outputs:            reference to double array of size P = 10.
	                    To be filled with the output node states.
	input_data:         reference to arma::vec of length M.
	                    Input vector. Contains the pixel values of an input image.
	input_weights:      reference to arma::mat of size N x (M + 1)
	                    Matrix W^in. Contains the weights connecting the input layer
					    to the first hidden layer (including the input bias weight).
	hidden_weights:     reference to arma::cube with L - 1 rows, N cols, N + 1 slices.
	                    Contains the weights between the hidden layers.
						1st index: layer (where the connection ends),
						2nd index: node index of layer where the connection ends,
						3rd index: node index of layer where the connections begins.
	output_weights:     reference to arma::mat with size P x (N + 1) (where P = 10).
	                    Contains the weights connecting the last hidden layer to the output layer.
	N:                  int.
						Nodes per hidden layer (except bias node).
	L:                  int.
						Number of hidden layers.
	*/
	
	double summe = 0.0;
	
	
	// We compute the activations of the first layer:
	for (int n = 0; n < N; ++n){
		summe = input_weights(n, M);  // bias weight
		for (int m = 0; m < M; ++m){
			summe += input_weights(n, m) * input_data(m);
		}
		g_primes(n) = input_processing_prime(summe);
		activations(0, n) = input_processing(summe);
	}	
	
	// compute node states for first hidden layer
	for (int n = 0; n < N; ++n){
		node_states(0, n) = f(activations(0, n));
	}
	
	// compute activations and node states for hidden layers 2, ..., L.
	// NOTE: hidden_weights contains weights for layers 2, ..., L,
	//       i.e. hidden_weights(0, n, j) is input to 2nd hidden layer,
	//            hidden_weights(1, n, j) is input to 3rd hidden layer etc.
	for (int l = 1; l < L; ++l){
		// compute activations
		for (int n = 0; n < N; ++n){
			summe = hidden_weights(l - 1, n, N);  // bias weight
			for (int j = 0; j < N; ++j){
				summe += hidden_weights(l - 1, n, j) * node_states(l - 1, j);
			}
			activations(l, n) = summe;
		}
		// compute node states
		for (int n = 0; n < N; ++n){
			node_states(l, n) = f(activations(l, n));
		}
	}
	
	// compute output activations
	for (int p = 0; p < P; ++p){
		summe = output_weights(p, N);  // bias weight
		for (int n = 0; n < N; ++n){
			summe += output_weights(p, n) * node_states(L - 1, n);
		}
		output_activations[p] = summe;
	}
	
	// compute outputs with clipping activation function
	for (int p = 0; p < P; ++p){
		if (output_activations[p] < 0.0){
			outputs[p] = 0.0;
		} else if (output_activations[p] > 1.0){
			outputs[p] = 1.0;
		} else {
			outputs[p] = output_activations[p];
		}
	}
}
