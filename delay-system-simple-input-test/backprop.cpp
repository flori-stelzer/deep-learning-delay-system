#include "backprop.h"

using namespace std;
using namespace arma;
using namespace globalconstants;



void get_deltas(mat &deltas, double (&output_deltas)[P], double (&outputs)[P], double (&targets)[P], mat &f_prime_activations, cube hidden_weights, mat output_weights, vector<int> &diag_indices, double theta, double alpha, int N, int L, vector<double> exp_table){
	/*
	Function to compute the deltas for backpropagation.
	
	Args:
	deltas: reference to armadillo matrix of size L x N.
	        To be filled with the deltas for the hidden layer nodes.
	output_deltas: reference to double array of length P = 10.
				   To be filled with the deltas for the output layer nodes.
	outputs: reference to double array of length P = 10.
	         Contains the outputs of the system.
	targets: reference to double array of length P = 10.
	         Contains the target which should be matched by the outputs.
	f_prime_activations: reference to armadillo matrix of size L x N.
	                     Contains the values of f' at the activations of the system (or network).
	hidden_weights: reference to armadillo cube of size (L - 1) x N x (N + 1).
	                Contains the hidden weights.
	output_weights: reference to armadillo matrix of size P x (N + 1), where P = 10.
	                Contains the output weights.
	diag_indices: reference to int vector of length D.
	              Contains the indices n'_d for the nonzero diagonals of the hidden weight matrices.
	theta: double.
	       Node Separation. theta = T / N.
	alpha: double.
	       Factor for the linear dynamics. Must be negative.
	N: int.
	   Nodes per hidden layer (except bias node).
	L: int.
	   Number of hidden layers.
	exp_table: double vector.
	           Contains the values exp(n * theta * alpha) for n = 0, 1, 2, 3, ...
	*/
	
	// vector to store the values of Delta_omega for each layer.
	vec delta_omega(L, fill::zeros);
	
	
	double exp_factor = exp(alpha * theta);
	double phi = (exp_factor - 1.0) / alpha;
	int exp_table_size = exp_table.size();
    
	// compute deltas for output layer:
	for (int p = 0; p < P; ++p){
		output_deltas[p] = outputs[p] - targets[p];
	}
	
	// compute deltas for last hidden layer:
	double outer_sum;
	double inner_sum;
	for (int n = 0; n < N; ++n){
		outer_sum = 0.0;
		for (int p = 0; p < P; ++p){
			inner_sum = 0.0;
			for (int j = n; j < N; ++j){
				if (j - n >= exp_table_size){
					break;
				}
				inner_sum += output_weights(p, j) * exp_table[j - n];
				//cout << p << " " << j << " : ow: " << output_weights(p, j) << "; exp: " << exp_table[j - n] << endl;
			}
			//cout << "is: " << inner_sum << endl;
			//cout << "od: " << output_deltas[p] << endl;
			outer_sum += output_deltas[p] * inner_sum;
		}
		//cout << "os: " << outer_sum << endl;
		deltas(L - 1, n) = phi * f_prime_activations(L - 1, n) * outer_sum;
	}
	// compute delta_omega for last hidden layer:
	outer_sum = 0.0;
	for (int p = 0; p < P; ++p){
		inner_sum = 0.0;
		for (int j = 0; j < N; ++j){
			if (j + 1 >= exp_table_size){
				break;
			}
			inner_sum += output_weights(p, j) * exp_table[j + 1];  // be careful: index count starts at zero (thats why j+1)
		}
		outer_sum += output_deltas[p] * inner_sum;
	}
	delta_omega(L - 1) = outer_sum;
	    
	// compute deltas for hidden layers L - 1, ..., 1
	for (int l = L - 2; l > -1; --l){
		for (int n = 0; n < N; ++n){
			outer_sum = 0.0;
			for (int i = 0; i < N; ++i){
				inner_sum = 0.0;
				
				for (int n_prime_d : diag_indices){
					int j = i - n_prime_d;
					if (j >= N){
						continue;
					}
					if (j - n >= exp_table_size){
						continue;
					}
					if (j < n){
						break;
					}
					inner_sum += hidden_weights(l, i, j) * exp_table[j - n];  // note l-index shift for hidden_weights
				}
				
				//for (int j = n; j < N; ++j){
				//	if (j - n >= exp_table_size){
				//		break;
				//	}
				//	inner_sum += hidden_weights(l, i, j) * exp_table[j - n];
				//}
				outer_sum += deltas(l + 1, i) * inner_sum;
			}
			if (N - 1 - n < exp_table_size){
				outer_sum += delta_omega(l + 1) * exp_table[N - 1 - n];
			}
			deltas(l, n) = phi * f_prime_activations(l, n) * outer_sum;
		}
		// compute delta_omega
		outer_sum = 0.0;
		for (int i = 0; i < N; ++i){
			inner_sum = 0.0;
			
			for (int n_prime_d : diag_indices){
				int j = i - n_prime_d;
				if (j >= N){
					continue;
				}
				if (j + 1 >= exp_table_size){
					continue;
				}
				if (j < 0){
					break;
				}
				inner_sum += hidden_weights(l, i, j) * exp_table[j + 1];  // be careful: index count starts at zero (thats why j+1)
			}
			
			//for (int j = 0; j < N; ++j){
			//	if (j + 1 >= exp_table_size){
			//		break;
			//	}
			//	inner_sum += hidden_weights(l, i, j) * exp_table[j + 1];  // be careful: index count starts at zero (thats why j+1)
			//}
			outer_sum += deltas(l + 1, i) * inner_sum;
		}
		delta_omega(l) = outer_sum;
	}
	
	/*
	for (int l = L - 2; l > -1; --l){
		for (int n = 0; n < N; ++n){
			double summe = 0;
			for (int i = 0; i < N; ++i){
				if (n == (N - 1) && i + 1 < exp_table_size){
					partial = exp_table[i + 1];  // be careful: index count starts at zero (thats why +1)
				} else {
					partial = 0.0;
				}
				for (int n_prime_d : diag_indices){
					int nu = n + n_prime_d;
					if (nu < 0){
						continue;
					}
					if (i - nu >= exp_table_size){
						continue;
					}
					if (nu > i){
						break;
					}
					partial += exp_table[i - nu] * phi * f_prime_activations(l + 1, nu) * hidden_weights(l, nu, n);  // note index shift for hidden_weights
				}
				summe += deltas(l + 1, i) * partial;
			}
			deltas(l, n) = summe;
		}
	}
	*/
}


void get_gradient(mat &input_weight_gradient, cube &weight_gradient, mat &output_weight_gradient, mat &deltas, double (&output_deltas)[P], vec &input_data, mat &node_states, mat &f_prime_activations, vec &g_primes, vector<int> diag_indices, double theta, double alpha, int N, int L, vector<double> exp_table){
	/*
	Function to compute the gradient.
	
	Args:
	input_weight_gradient: reference to arma::mat of size N x (M + 1).
	                       To be filled with partial derivatives w.r.t. input weights.
	weight_gradient: reference to arma::cube of size (L - 1) x N x (N + 1).
	                 Initially filled with zeros.
	                 To be filled with partial derivatives w.r.t. hidden weights.
					 For constant zero weights, the corresponding entry of the cube must stay zero.
	output_weight_gradient: reference to arma::mat of size P x (N + 1).
							To be filled with partial derivatives w.r.t. output weights.
	deltas: reference to arma::mat of size L x N.
	        The delta matrix obtained from the get_deltas function.
	output_deltas: reference to double array of length P = 10.
	               The deltas for the output nodes obtained from the get_deltas function.
	input_data: reference to arma::vec of length M = 784.
	            Input vector. Contains the pixel values of an input image.
	node_states: reference to arma::mat with size L x N.
	             Contains states of the hidden nodes.
	f_prime_activations: reference to armadillo matrix of size L x N.
	                     Contains the values of f' at the activations of the system (or network).
	diag_indices: reference to int vector of length D.
	              Contains the indices n'_d for the nonzero diagonals of the hidden weight matrices.
	theta: double.
	       Node Separation. theta = T / N.
	alpha: double.
	       Factor for the linear dynamics. Must be negative.
	N: int.
	   Nodes per hidden layer (except bias node).
	L: int.
	   Number of hidden layers.
	exp_table: double vector.
	           Contains the values exp(n * theta * alpha) for n = 0, 1, 2, 3, ...
	*/
	
	
	// input weight gradients
	for (int n = 0; n < N; ++n){
		for (int m = 0; m < M; ++m){
			input_weight_gradient(n, m) = deltas(0, n) * g_primes(n) * input_data(m);
		}
		input_weight_gradient(n, M) = deltas(0, n) * g_primes(n);
	}
	
	
	// inner weights gradients
	// only on diagonals and last column
	for (int l = 1; l < L; ++l){
		for (int i = 0; i < N; ++i){
			// diagonals
			for (int n_prime_d : diag_indices){
				int j = i - n_prime_d;
				if (j >= N){
					continue;
				}
				if (j < 0){
					break;
				}
				weight_gradient(l - 1, i, j) = deltas(l, i) * node_states(l - 1, j);
			}
			// last column, j = N + 1
			weight_gradient(l - 1, i, N) = deltas(l, i);
		}
	}
	
	// output weight gradients
	for (int p = 0; p < P; ++p){
		for (int n = 0; n < N; ++n){
			output_weight_gradient(p, n) = output_deltas[p] * node_states(L - 1, n);
		}
		output_weight_gradient(p, N) = output_deltas[p];  // for bias output weight
	}
}


void get_gradient_classical_backprop(mat &input_weight_gradient, cube &weight_gradient, mat &output_weight_gradient, vec &input_data, mat &node_states, mat &f_prime_activations, vec &g_primes, double (&outputs)[P], double (&targets)[P], cube &hidden_weights, mat &output_weights, vector<int> diag_indices, int N, int L){
	/*
	Function to compute the gradient using classical backpropagation.
	For the hidden weights only the gradients for the nonzero diagonals (and the bias weights) must be nonzero. 
	
	Args:
	input_weight_gradient: reference to arma::mat of size N x (M + 1).
	                       To be filled with partial derivatives w.r.t. input weights.
	weight_gradient: reference to arma::cube of size (L - 1) x N x (N + 1).
	                 Initially filled with zeros.
	                 To be filled with partial derivatives w.r.t. hidden weights.
					 For constant zero weights, the corresponding entry of the cube must stay zero.
	output_weight_gradient: reference to arma::mat of size P x (N + 1).
							To be filled with partial derivatives w.r.t. output weights.
	input_data: reference to arma::vec of length M = 784.
	            Input vector. Contains the pixel values of an input image.
	node_states: reference to arma::mat with size L x N.
	             Contains states of the hidden nodes.
	f_prime_activations: reference to armadillo matrix of size L x N.
	                     Contains the values of f' at the activations of the system (or network).
	outputs: reference to double array of length P = 10.
	         Contains the outputs of the system.
	targets: reference to double array of length P = 10.
	         Contains the target which should be matched by the outputs.
	hidden_weights: reference to armadillo cube of size (L - 1) x N x (N + 1).
	                Contains the hidden weights.
	output_weights: reference to armadillo matrix of size P x (N + 1), where P = 10.
	                Contains the output weights.
	diag_indices: reference to int vector of length D.
	              Contains the indices n'_d for the nonzero diagonals of the hidden weight matrices.
	N: int.
	   Nodes per hidden layer (except bias node).
	L: int.
	   Number of hidden layers.
	*/
	
	
	// get "classical" deltas
	
	// arrays to store deltas:
	mat deltas(L, N);
	double output_deltas[P];
	
	// compute deltas for output layer:
	for (int p = 0; p < P; ++p){
		output_deltas[p] = outputs[p] - targets[p];
	}
	
	// compute deltas for last hidden layer:
	for (int n = 0; n < N; ++n){
		double summe = 0;
		for (int p = 0; p < P; ++p){
			summe += output_deltas[p] * output_weights(p, n);
		}
		deltas(L - 1, n) = f_prime_activations(L - 1, n) * summe;
	}
	
	// compute deltas for hidden layer L - 1, ..., 1:
	for (int l = L - 2; l > 0; --l){
		for (int n = 0; n < N; ++n){
			double summe = 0;
			for (int n_prime_d : diag_indices){
				int i = n + n_prime_d;
				if (i < 0){
					continue;
				}
				if (i >= N){
					break;
				}
				summe += deltas(l + 1, i) * hidden_weights(l, i, n);
			//	summe += deltas(l + 1, i) * hidden_weights(l, i, n);
			//for (int i = 0; i < N; ++i){
			//	summe += deltas(l + 1, i) * hidden_weights(l, i, n);
			}
			deltas(l, n) = f_prime_activations(l, n) * summe;
		}
	}
	
	// input weight gradient
	/*
	for (int n = 0; n < N; ++n){
		for (int m = 0; m < M; ++m){
			input_weight_gradient(n, m) = deltas(0, n) * input_data(m)  * g_primes(n);
		}
		// for m = M + 1
		input_weight_gradient(n, M) = deltas(0, n)  * g_primes(n);
	}
	*/
	
	// hidden weight gradient
	for (int l = 1; l < L; ++l){
		for (int n = 0; n < N; ++n){
			for (int n_prime_d : diag_indices){
				int j = n - n_prime_d;
				if (j >= N){
					continue;
				}
				if (j < 0){
					break;
				}
				weight_gradient(l - 1, n, j) = deltas(l, n) * node_states(l - 1, j);
			}
			// for j = N + 1
			weight_gradient(l - 1, n, N) = deltas(l, n);
		}
	}
		
	// output weight gradient
	for (int p = 0; p < P; ++p){
		for (int n = 0; n < N; ++n){
			output_weight_gradient(p, n) = output_deltas[p] * node_states(L - 1, n);
		}
		output_weight_gradient(p, N) = output_deltas[p];  // for bias output weight
	}
}

