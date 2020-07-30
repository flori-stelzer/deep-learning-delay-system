#ifndef SOLVE_DDE_H
#define SOLVE_DDE_H

#include <iostream>
#include <armadillo>

#include "f.h"
#include "global_constants.h"

void solve_dde_heun(arma::mat &activations, arma::mat &node_states, double (&output_activations)[globalconstants::P], double (&outputs)[globalconstants::P], arma::vec &g_primes,
			   arma::vec &input_data, arma::mat input_weights, arma::cube hidden_weights, arma::mat output_weights, std::vector<int> &diag_indices, double theta, double alpha,
			   int N, int L, int N_h);
void solve_dde_ibp(arma::mat &activations, arma::mat &node_states, double (&output_activations)[globalconstants::P], double (&outputs)[globalconstants::P], arma::vec &g_primes,
			   arma::vec &input_data, arma::mat input_weights, arma::cube hidden_weights, arma::mat output_weights, std::vector<int> &diag_indices, double theta, double alpha,
			   int N, int L, int N_h, int record_example=0, int rec_step=0);
#endif
