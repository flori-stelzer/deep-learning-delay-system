#include "solve_dde.h"

using namespace std;
using namespace arma;
using namespace globalconstants;



void solve_dde_heun(mat &activations, mat &node_states, double (&output_activations)[P], double (&outputs)[P], vec &g_primes,
				   vec &input_data, mat input_weights, cube hidden_weights, mat output_weights, vector<int> &diag_indices, double theta, double alpha,
				   int N, int L, int N_h)
{
	/*
	Function to solve the delay differential equations by a semi-analytic Heun method.
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
	diag_indices:       reference to int vector of length D.
	                    Contains the indices n'_d for the nonzero diagonals of the hidden weight matrices.
	theta:              double.
	                    Node Separation. theta = T / N.
	alpha:              double.
	                    Factor for the linear dynamics. Must be negative.
	N:                  int.
						Nodes per hidden layer (except bias node).
	L:                  int.
						Number of hidden layers.
	N_h:                int.
						computation step lenght h = theta / N_h.
	*/
	
	
	double x_0 = 0.0;
	double summe;
	double h = theta / (double)N_h;
	double exp_factor_h = exp(alpha * h);
	
	// In the following we often need the values exp(alpha * n_h * h). That's why we make a lookup table.
	vector<double> exp_h_table;
	for (int n_h = 0; n_h < N_h + 1; ++n_h){
		double exp_h_value = exp(alpha * h * (double)n_h);
		exp_h_table.push_back(exp_h_value);
	}
	
	
	// We need to memorize the values of and a^\ell_{n, n_h} and x^\ell_{n, n_h}, but only for one layer at a time.
	// We also need a matrix to save the activated states f( a^\ell_{n, n_h} ) to accelerate the computation.
	// Thus, we use armadillo matrices:
	mat a_states(N, N_h + 1);
	mat fa_states(N, N_h +1);
	mat x_states(N, N_h);
	
	
	// We compute the activations of the first layer:
	for (int n = 0; n < N; ++n){
		summe = input_weights(n, M);  // bias weight
		for (int m = 0; m < M; ++m){
			summe += input_weights(n, m) * input_data(m);
		}
		g_primes(n) = input_processing_prime(summe);
		activations(0, n) = input_processing(summe);
	}
	
	
	// For the first layer, we can calculate the exact values of x.
	// We begin with the x values up to the first node:
	for (int n_h = 0; n_h < N_h; ++n_h){
		x_states(0, n_h) = exp_h_table[n_h + 1] * x_0 + 1.0/alpha * (exp_h_table[n_h + 1] - 1.0) * f(activations(0, 0));
	}
	// Now we know the value x(theta) of the first node of the first layer:
	node_states(0, 0) = x_states(0, N_h - 1);
	// We continue with the calculation of the remaining x values for the first layer:
	for (int n = 1; n < N; ++n){
		for (int n_h = 0; n_h < N_h; ++n_h){
			x_states(n, n_h) = exp_h_table[n_h + 1] * x_states(n - 1, N_h - 1) + 1.0/alpha * (exp_h_table[n_h + 1] - 1.0) * f(activations(0, n));
		}
		// From the x_states of the first layer, we save the values of the node states:
		node_states(0, n) = x_states(n, N_h - 1);
	}
	
	
	// For the other layers, we need to use an approximation rule:
	// Mind the l-index shift in the hidden_weights array!
	for (int l = 1; l < L; ++l){
		// first we get the a_states for the current layer:
		for (int n = 0; n < N; ++n){
			// a_states for n_h = 0:
			// bias weight + first summand:
			if (l == 1){
				summe = hidden_weights(l - 1, n, N) + hidden_weights(l - 1, n, 0) * x_0;
			} else {
				summe = hidden_weights(l - 1, n, N) + hidden_weights(l - 1, n, 0) * node_states(l - 2, N - 1);
			}
			// other summands:
			for (int n_prime_d : diag_indices){
				int j = n - n_prime_d;
				if (j >= N){
					continue;
				}
				if (j < 1){
					break;
				}
				summe += hidden_weights(l - 1, n, j) * node_states(l - 1, j - 1);
			}
			a_states(n, 0) = summe;
			// a_states for other n_h:
			for (int n_h = 0; n_h < N_h; ++n_h){
				summe = hidden_weights(l - 1, n, N);
				for (int n_prime_d : diag_indices){
					int j = n - n_prime_d;
					if (j >= N){
						continue;
					}
					if (j < 0){
						break;
					}
					summe += hidden_weights(l - 1, n, j) * x_states(j, n_h);
				}
				a_states(n, n_h + 1) = summe;
			}
			// the last a_state on a theta-interval is the activation:
			activations(l, n) = a_states(n, N_h);
		}
		// get f(a):
		fa_states = f_matrix(a_states);
		// Now we can compute the x_states for the current layer:
		// first case: n = 1 and n_h = 1:
		x_states(0, 0) = exp_factor_h * node_states(l - 1, N - 1) + 0.5 * h * (exp_factor_h * fa_states(0, 0) + fa_states(0, 1));
		// second case: n = 1 and n_h > 1:
		for (int n_h = 1; n_h < N_h; ++n_h){
			x_states(0, n_h) = exp_factor_h * x_states(0, n_h - 1) + 0.5 * h * (exp_factor_h * fa_states(0, n_h) + fa_states(0, n_h + 1));
		}
		// node state is x_state on theta-grid-point:
		node_states(l, 0) = x_states(0, N_h - 1);
		for (int n = 1; n < N; ++n){
			// third case: n > 1 and n_h = 1:
			x_states(n, 0) = exp_factor_h * x_states(n - 1, N_h - 1) + 0.5 * h * (exp_factor_h * fa_states(n, 0) + fa_states(n, 1));
			// fourth case: n > 1 and n_h > 1:
			for (int n_h = 1; n_h < N_h; ++n_h){
				x_states(n, n_h) = exp_factor_h * x_states(n, n_h - 1) + 0.5 * h * (exp_factor_h * fa_states(n, n_h) + fa_states(n, n_h + 1));
			}
			// node state is x_state on theta-grid-point:
			node_states(l, n) = x_states(n, N_h - 1);
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
	
	// compute outputs with softmax function
	double exp_sum = 0;
	for (int p = 0; p < P; ++p){
		exp_sum += exp(output_activations[p]);
	}
	for (int p = 0; p < P; ++p){
		outputs[p] = exp(output_activations[p])/exp_sum;
	}
}


void solve_dde_ibp(mat &activations, mat &node_states, double (&output_activations)[P], double (&outputs)[P], vec &g_primes,
				   vec &input_data, mat input_weights, cube hidden_weights, mat output_weights, vector<int> &diag_indices, double theta, double alpha,
				   int N, int L, int N_h, int record_example, int rec_step)
{
	/*
	Function to solve the delay differential equations by another variant of the semi-analytic Heun method
	employing integration by parts.
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
	diag_indices:       reference to int vector of length D.
	                    Contains the indices n'_d for the nonzero diagonals of the hidden weight matrices.
	theta:              double.
	                    Node Separation. theta = T / N.
	alpha:              double.
	                    Factor for the linear dynamics. Must be negative.
	N:                  int.
						Nodes per hidden layer (except bias node).
	L:                  int.
						Number of hidden layers.
	N_h:                int.
						computation step lenght h = theta / N_h.
	record_example:		int.
						Parameter for data recording for video.
	rec_step:			int.
						Parameter for data recording for video.
	*/
	
	
	double x_0 = 0.0;
	double summe;
	double h = theta / (double)N_h;
	double exp_factor_h = exp(alpha * h);
	double phi_h = (exp_factor_h - 1.0) / (pow(alpha, 2.0) * h);
	
	// In the following we often need the values exp(alpha * n_h * h). That's why we make a lookup table.
	vector<double> exp_h_table;
	for (int n_h = 0; n_h < N_h + 1; ++n_h){
		double exp_h_value = exp(alpha * h * (double)n_h);
		exp_h_table.push_back(exp_h_value);
	}
	
	
	// We need to memorize the values of and a^\ell_{n, n_h} and x^\ell_{n, n_h}, but only for one layer at a time.
	// We also need a matrix to save the activated states f( a^\ell_{n, n_h} ) to accelerate the computation.
	// Thus, we use armadillo matrices:
	mat a_states(N, N_h + 1);
	mat fa_states(N, N_h +1);
	mat x_states(N, N_h);
	
	
	// We compute the activations of the first layer:
	for (int n = 0; n < N; ++n){
		summe = input_weights(n, M);  // bias weight
		for (int m = 0; m < M; ++m){
			summe += input_weights(n, m) * input_data(m);
		}
		g_primes(n) = input_processing_prime(summe);
		activations(0, n) = input_processing(summe);
	}
	
	
	// For the first layer, we can calculate the exact values of x.
	// We begin with the x values up to the first node: 
	for (int n_h = 0; n_h < N_h; ++n_h){
		x_states(0, n_h) = exp_h_table[n_h + 1] * x_0 + 1.0/alpha * (exp_h_table[n_h + 1] - 1.0) * f(activations(0, 0));
	}
	// Now we know the value x(theta) of the first node of the first layer:
	node_states(0, 0) = x_states(0, N_h - 1);
	// We continue with the calculation of the remaining x values for the first layer:
	for (int n = 1; n < N; ++n){
		for (int n_h = 0; n_h < N_h; ++n_h){
			x_states(n, n_h) = exp_h_table[n_h + 1] * x_states(n - 1, N_h - 1) + 1.0/alpha * (exp_h_table[n_h + 1] - 1.0) * f(activations(0, n));
		}
		// From the x_states of the first layer, we save the values of the node states:
		node_states(0, n) = x_states(n, N_h - 1);
	}
	
	// save x-states for the current layer:
	if (record_example > 0){
		string x_file_path = "video/x_states_step_" + to_string(rec_step) + "_example_" + to_string(record_example) + "_layer_1.txt";
		x_states.save(x_file_path, csv_ascii);
	}
	
	
	// For the other layers, we need to use an approximation rule:
	// Mind the l-index shift in the hidden_weights array!
	for (int l = 1; l < L; ++l){
		// first we get the a_states for the current layer:
		for (int n = 0; n < N; ++n){
			// a_states for n_h = 0:
			// bias weight + first summand:
			if (l == 1){
				summe = hidden_weights(l - 1, n, N) + hidden_weights(l - 1, n, 0) * x_0;
			} else {
				summe = hidden_weights(l - 1, n, N) + hidden_weights(l - 1, n, 0) * node_states(l - 2, N - 1);
			}
			// other summands:
			for (int n_prime_d : diag_indices){
				int j = n - n_prime_d;
				if (j >= N){
					continue;
				}
				if (j < 1){
					break;
				}
				summe += hidden_weights(l - 1, n, j) * node_states(l - 1, j - 1);
			}
			a_states(n, 0) = summe;
			// a_states for other n_h:
			for (int n_h = 0; n_h < N_h; ++n_h){
				summe = hidden_weights(l - 1, n, N);
				for (int n_prime_d : diag_indices){
					int j = n - n_prime_d;
					if (j >= N){
						continue;
					}
					if (j < 0){
						break;
					}
					summe += hidden_weights(l - 1, n, j) * x_states(j, n_h);
				}
				a_states(n, n_h + 1) = summe;
			}
			// the last a_state on a theta-interval is the activation:
			activations(l, n) = a_states(n, N_h);
		}
		// get f(a):
		fa_states = f_matrix(a_states);
		// Now we can compute the x_states for the current layer:
		// first case: n = 1 and n_h = 1:
		x_states(0, 0) = exp_factor_h * node_states(l - 1, N - 1) + 1.0/alpha * (exp_factor_h * fa_states(0, 0) - fa_states(0, 1)) - phi_h * (fa_states(0, 0) - fa_states(0, 1));
		// second case: n = 1 and n_h > 1:
		for (int n_h = 1; n_h < N_h; ++n_h){
			x_states(0, n_h) = exp_factor_h * x_states(0, n_h - 1) + 1.0/alpha * (exp_factor_h * fa_states(0, n_h) - fa_states(0, n_h + 1)) - phi_h * (fa_states(0, n_h) - fa_states(0, n_h + 1));
		}
		// node state is x_state on theta-grid-point:
		node_states(l, 0) = x_states(0, N_h - 1);
		for (int n = 1; n < N; ++n){
			// third case: n > 1 and n_h = 1:
			x_states(n, 0) = exp_factor_h * x_states(n - 1, N_h - 1) + 1.0/alpha * (exp_factor_h * fa_states(n, 0) - fa_states(n, 1)) - phi_h * (fa_states(n, 0) - fa_states(n, 1));
			// fourth case: n > 1 and n_h > 1:
			for (int n_h = 1; n_h < N_h; ++n_h){
				x_states(n, n_h) = exp_factor_h * x_states(n, n_h - 1) + 1.0/alpha * (exp_factor_h * fa_states(n, n_h) - fa_states(n, n_h + 1)) - phi_h * (fa_states(n, n_h) - fa_states(n, n_h + 1));
			}
			// node state is x_state on theta-grid-point:
			node_states(l, n) = x_states(n, N_h - 1);
		}
		
		// save x-states for the current layer:
		if (record_example > 0){
			string x_file_path = "video/x_states_step_" + to_string(rec_step) + "_example_" + to_string(record_example) + "_layer_" + to_string(l + 1) + ".txt";
			x_states.save(x_file_path, csv_ascii);
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
	
	// compute outputs with softmax function
	double exp_sum = 0;
	for (int p = 0; p < P; ++p){
		exp_sum += exp(output_activations[p]);
	}
	for (int p = 0; p < P; ++p){
		outputs[p] = exp(output_activations[p])/exp_sum;
	}
}
