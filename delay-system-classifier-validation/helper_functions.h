#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <armadillo>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <ctime>

#include "global_constants.h"

extern std::mt19937 rng;
double uniform(double a, double b);
int uniform_int(int a, int b);
double learning_rate(int epoch, int k, double eta_0, double eta_1);
void read_files(arma::cube &train_images, arma::mat &test_images, int (&train_labels)[globalconstants::number_of_training_batches][globalconstants::training_batch_size],int (&test_labels)[globalconstants::test_batch_size], std::string data_dir);
void initialize_weights(arma::mat &input_weights, arma::cube &hidden_weights, arma::mat &output_weights, int D, int L, int N, std::vector<int> diag_indices, double initial_input_weigt_radius, double initial_hidden_weigt_radius, double initial_output_weigt_radius, bool save_to_file);
void load_initial_weights(arma::mat &input_weights, arma::cube &hidden_weights, arma::mat &output_weights);
std::vector<int> get_diag_indices(int N, int D, std::string method, int diag_distance, std::string diag_file_path, int line_index);
void get_targets(double (&targets)[globalconstants::P], int label);
void pixel_shift28(arma::vec &input_data, int max_pixel_shift);
void pixel_shift32(arma::vec &input_data, int max_pixel_shift);
void bilinear_interpolation32(arma::vec &input_data, arma::cube vectorfield);
void rotation32(arma::vec &input_data, double rotation_angle);
void horizontal_flip32(arma::vec &input_data);
void print_parameters(std::string results_file_name, std::string print_msg, std::string task, bool cross_validation, std::string system_simu, std::string grad_comp, bool gradient_check, int N, int L, int D, double theta, double alpha, std::string diag_method, int diag_distance, std::string diag_file_path, int number_of_epochs, double eta_0, double eta_1, bool pixel_shift, bool input_noise, double training_noise_sigma, int N_h, double exp_precision);
void print_results(std::string results_file_name, int validation_batch_index, std::vector<int> diag_indices, std::vector<double> training_accuracy_vector, std::vector<double> accuracy_vector, std::vector<double> similarity_vector);
void print_weights(arma::mat input_weights, arma::cube hidden_weights, arma::mat output_weights, std::vector<int> diag_indices, int validation_batch_index, int epoch);

#endif  
