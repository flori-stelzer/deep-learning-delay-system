#ifndef NUMERICAL_GRADIENT_H
#define NUMERICAL_GRADIENT_H

#include <iostream>
#include <armadillo>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>

#include "solve_dde.h"
#include "global_constants.h"

void get_num_gradient(arma::mat &num_input_weight_gradient, arma::cube &num_weight_gradient, arma::mat &num_output_weight_gradient,
					  arma::mat &activations, arma::mat &node_states, double (&output_activations)[globalconstants::P], double (&outputs)[globalconstants::P], arma::vec &g_primes, int label,
					  arma::vec &input_data, arma::mat &input_weights, arma::cube &hidden_weights, arma::mat &output_weights, std::vector<int> &diag_indices, double theta, double alpha, int N, int L, int N_h, double epsilon);
void write_gradients_to_file(arma::mat &input_weight_gradient, arma::cube &weight_gradient, arma::mat &output_weight_gradient, arma::mat &num_input_weight_gradient, arma::cube &num_weight_gradient, arma::mat &num_output_weight_gradient, int epoch, int step_index, int validation_batch_index);
double cosine_similarity(arma::mat &input_weight_gradient, arma::cube &weight_gradient, arma::mat &output_weight_gradient, arma::mat &num_input_weight_gradient, arma::cube &num_weight_gradient, arma::mat &num_output_weight_gradient);


#endif
