# include "numerical_gradient.h"

using namespace std;
using namespace arma;
using namespace globalconstants;


void get_num_gradient(mat &num_input_weight_gradient, cube &num_weight_gradient, mat &num_output_weight_gradient,
					  mat &activations, mat &node_states, double (&output_activations)[P], double (&outputs)[P], vec &g_primes, int label,
					  vec &input_data, mat &input_weights, cube &hidden_weights, mat &output_weights, vector<int> &diag_indices, double theta, double alpha, int N, int L, int N_h, double epsilon){
	/*
	Function computes the gradient numerically. So we check whether the backpropagation algorithm works correctly.
	
	Parameter:
	...
	*/
	
	// numerical partial derivatives for input layer
	for (int n = 0; n < N; ++n){
		for (int m = 0; m < M + 1; ++m){
			mat input_weights_plus = input_weights;
			input_weights_plus(n, m) += epsilon;
			double outputs_plus[P];
			solve_dde_ibp(activations, node_states, output_activations, outputs_plus, g_primes,
							input_data, input_weights_plus, hidden_weights, output_weights, diag_indices, theta, alpha,
							N, L, N_h);
			double loss_plus = - log(outputs_plus[label]);
			mat input_weights_minus = input_weights;
			input_weights_minus(n, m) -= epsilon;
			double outputs_minus[P];
			solve_dde_ibp(activations, node_states, output_activations, outputs_minus, g_primes,
							input_data, input_weights_minus, hidden_weights, output_weights, diag_indices, theta, alpha,
							N, L, N_h);
			double loss_minus = - log(outputs_minus[label]);
			num_input_weight_gradient(n, m) = 0.5 * (loss_plus - loss_minus)/epsilon;
		}
	}
	
	// numerical partial derivatives for hidden layers
	for (int l = 0; l < L - 1; ++l){
		for (int n = 0; n < N; ++n){
			// diagonals:
			for (int n_prime : diag_indices){
				int j = n - n_prime;
				if (j >= N){
					continue;
				}
				if (j < 0){
					break;
				}
				cube hidden_weights_plus = hidden_weights;
				hidden_weights_plus(l, n, j) += epsilon;
				double outputs_plus[P];
				solve_dde_ibp(activations, node_states, output_activations, outputs_plus, g_primes,
							input_data, input_weights, hidden_weights_plus, output_weights, diag_indices, theta, alpha,
							N, L, N_h);
				double loss_plus = - log(outputs_plus[label]);
				cube hidden_weights_minus = hidden_weights;
				hidden_weights_minus(l, n, j) -= epsilon;
				double outputs_minus[P];
				solve_dde_ibp(activations, node_states, output_activations, outputs_minus, g_primes,
							input_data, input_weights, hidden_weights_minus, output_weights, diag_indices, theta, alpha,
							N, L, N_h);
				double loss_minus = - log(outputs_minus[label]);
				num_weight_gradient(l, n, j) = 0.5 * (loss_plus - loss_minus)/epsilon;
			}
			// last column:
			cube hidden_weights_plus = hidden_weights;
			hidden_weights_plus(l, n, N) += epsilon;
			double outputs_plus[P];
			solve_dde_ibp(activations, node_states, output_activations, outputs_plus, g_primes,
						input_data, input_weights, hidden_weights_plus, output_weights, diag_indices, theta, alpha,
						N, L, N_h);
			double loss_plus = - log(outputs_plus[label]);
			cube hidden_weights_minus = hidden_weights;
			hidden_weights_minus(l, n, N) -= epsilon;
			double outputs_minus[P];
			solve_dde_ibp(activations, node_states, output_activations, outputs_minus, g_primes,
						input_data, input_weights, hidden_weights_minus, output_weights, diag_indices, theta, alpha,
						N, L, N_h);
			double loss_minus = - log(outputs_minus[label]);
			num_weight_gradient(l, n, N) = 0.5 * (loss_plus - loss_minus)/epsilon;
		}
	}
	
	// numerical partial derivatives for output layer
	for (int p = 0; p < P; ++p){
		for (int n = 0; n < N + 1; ++n){
			mat output_weights_plus = output_weights;
			output_weights_plus(p, n) += epsilon;
			double outputs_plus[P];
			solve_dde_ibp(activations, node_states, output_activations, outputs_plus, g_primes,
							input_data, input_weights, hidden_weights, output_weights_plus, diag_indices, theta, alpha,
							N, L, N_h);
			double loss_plus = - log(outputs_plus[label]);
			mat output_weights_minus = output_weights;
			output_weights_minus(p, n) -= epsilon;
			double outputs_minus[P];
			solve_dde_ibp(activations, node_states, output_activations, outputs_minus, g_primes,
							input_data, input_weights, hidden_weights, output_weights_minus, diag_indices, theta, alpha,
							N, L, N_h);
			double loss_minus = - log(outputs_minus[label]);
			num_output_weight_gradient(p, n) = 0.5 * (loss_plus - loss_minus)/epsilon;
		}
	}
}


void write_gradients_to_file(mat &input_weight_gradient, cube &weight_gradient, mat &output_weight_gradient, mat &num_input_weight_gradient, cube &num_weight_gradient, mat &num_output_weight_gradient, int epoch, int step_index, int validation_batch_index){
	/*
	Function to save gradient and numerical gradient to text files in the folder "gradients".
	*/
	
	input_weight_gradient.save("gradients/input_weight_gradient_" + to_string(validation_batch_index) + "_" + to_string(epoch) + "_" + to_string(step_index) + ".txt", arma_ascii);
	weight_gradient.save("gradients/hidden_weight_gradient_" + to_string(validation_batch_index) + "_" + to_string(epoch) + "_" + to_string(step_index) + ".txt", arma_ascii);
	output_weight_gradient.save("gradients/output_weight_gradient_" + to_string(validation_batch_index) + "_" + to_string(epoch) + "_" + to_string(step_index) + ".txt", arma_ascii);
	
	num_input_weight_gradient.save("gradients/input_weight_gradient_numerical_" + to_string(validation_batch_index) + "_" + to_string(epoch) + "_" + to_string(step_index) + ".txt", arma_ascii);
	num_weight_gradient.save("gradients/hidden_weight_gradient_numerical_" + to_string(validation_batch_index) + "_" + to_string(epoch) + "_" + to_string(step_index) + ".txt", arma_ascii);
	num_output_weight_gradient.save("gradients/output_weight_gradient_numerical_" + to_string(validation_batch_index) + "_" + to_string(epoch) + "_" + to_string(step_index) + ".txt", arma_ascii);
}


double cosine_similarity(mat &input_weight_gradient, cube &weight_gradient, mat &output_weight_gradient, mat &num_input_weight_gradient, cube &num_weight_gradient, mat &num_output_weight_gradient){
	/*
	Function to compute cosine similarity of backpropagation gradient and numerical gradient.
	*/
	
	double scalar_product = accu(input_weight_gradient % num_input_weight_gradient) + accu(weight_gradient % num_weight_gradient) + accu(output_weight_gradient % num_output_weight_gradient);
	double squared_norm_bp = accu(input_weight_gradient % input_weight_gradient) + accu(weight_gradient % weight_gradient) + accu(output_weight_gradient % output_weight_gradient);
	double squared_norm_num = accu(num_input_weight_gradient % num_input_weight_gradient) + accu(num_weight_gradient % num_weight_gradient) + accu(num_output_weight_gradient % num_output_weight_gradient);
	double cos_similarity = scalar_product / (sqrt(squared_norm_bp) * sqrt(squared_norm_num));
	
	return cos_similarity;
}