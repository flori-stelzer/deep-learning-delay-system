#ifndef BACKPROP_H
#define BACKPROP_H

#include <iostream>
#include <armadillo>

#include "global_constants.h"

void get_deltas(arma::mat &deltas, double (&output_deltas)[globalconstants::P], double (&outputs)[globalconstants::P], double (&targets)[globalconstants::P], arma::mat &f_prime_activations, arma::cube hidden_weights, arma::mat output_weights, std::vector<int> &diag_indices, double theta, double alpha, int N, int L, std::vector<double> exp_table);
void get_gradient(arma::mat &input_weight_gradient, arma::cube &weight_gradient, arma::mat &output_weight_gradient, arma::mat &deltas, double (&output_deltas)[globalconstants::P], arma::vec &input_data, arma::mat &node_states, arma::mat &f_prime_activations, arma::vec &g_primes, std::vector<int> diag_indices, double theta, double alpha, int N, int L, std::vector<double> exp_table);
void get_gradient_classical_backprop(arma::mat &input_weight_gradient, arma::cube &weight_gradient, arma::mat &output_weight_gradient, arma::vec &input_data, arma::mat &node_states, arma::mat &f_prime_activations, arma::vec &g_primes, double (&outputs)[globalconstants::P], double (&targets)[globalconstants::P], arma::cube hidden_weights, arma::mat output_weights, std::vector<int> diag_indices, int N, int L);

#endif  
