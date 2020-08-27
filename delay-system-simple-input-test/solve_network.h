#ifndef SOLVE_NETWORK_H
#define SOLVE_NETWORK_H

#include <iostream>
#include <armadillo>

#include "f.h"
#include "global_constants.h"

void solve_network(arma::mat &activations, arma::mat &node_states, double (&output_activations)[globalconstants::P], double (&outputs)[globalconstants::P], arma::vec &g_primes,
				   arma::vec &input_data, arma::mat input_weights, arma::cube hidden_weights, arma::mat output_weights, double theta, double alpha,
				  int N, int L);
void solve_network_decoupled(arma::mat &activations, arma::mat &node_states, double (&output_activations)[globalconstants::P], double (&outputs)[globalconstants::P], arma::vec &g_primes,
				   arma::vec &input_data, arma::mat input_weights, arma::cube hidden_weights, arma::mat output_weights, int N, int L);
void solve_network_decoupled_fast(arma::mat &activations, arma::mat &node_states, double (&output_activations)[globalconstants::P], double (&outputs)[globalconstants::P], arma::vec &g_primes,
				   arma::vec &input_data, arma::mat &input_weights, arma::cube &hidden_weights, arma::mat &output_weights, int N, int L, std::vector<int> &diag_indices);


#endif
