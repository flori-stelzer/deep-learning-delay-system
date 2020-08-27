#include <iostream>
#include <armadillo>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include "helper_functions.h"
#include "solve_network.h"
#include "solve_dde.h"
#include "backprop.h"
#include "numerical_gradient.h"
#include "katana_get_params.hh"
#include "global_constants.h"


using namespace std;
using namespace arma;
using namespace globalconstants;

int main(int argc, char const *argv[])
{
	// for cpu time measuring
	clock_t start_overall = clock();
	double cumulative_solve_time = 0.0;
	double cumulative_backprop_time = 0.0;
	
	arma_rng::set_seed_random();
	
	
	// ### ### ### --- OPTIONS --- ### ### ###
	
	// file name of the text file where parameters and results will be saved
	string results_file_name = katana::getCmdOption(argv, argv + argc, "-filename", "results.txt");
	// print message to results file
	string print_msg = "Simulation of the deep learning delay system.";
	
	// task, possible options are "MNIST", "Fashion-MNIST", "CIFAR-10", "SVHN"
	string task = katana::getCmdOption(argv, argv + argc, "-task", "CIFAR-10");
	// modify global_constants.h accordingly!
	
	// print weights (and diagonals) after each training epoch to text files in weights folder?
	bool print_weights_to_file = false;
	
	// record data for time-signal video (or plot), only for MNIST or Fashion-MNIST
	bool record_data = katana::getCmdOption_bool(argv, argv + argc, "-record_data", false);
	int number_of_examples = katana::getCmdOption(argv, argv + argc, "-record_examples", 10);
	
	// If the following option is true, the program will not train the machine learning system.
	// Instead it will just choose diag indices n_prime according to the given paramters
	// and save them in the text file diag.txt. This will be done 6 times, i.e. the text file
	// will contain 6 lines of diag indices, which can be used for 6-fold cross validation
	bool make_diags = katana::getCmdOption_bool(argv, argv + argc, "-make_diags", false);
	// If the following option is true, the program will create initial weight files and stop.
	bool save_init_weights = katana::getCmdOption_bool(argv, argv + argc, "-save_init_weights", false);
	// If the following option is true, the program will load the initial weights from files instead of creating random weights.
	// Note that the corresponding diag file must be loaded as well.
	bool init_weights_from_file = katana::getCmdOption_bool(argv, argv + argc, "-init_weights_from_file", false);
	
	
	// option for system simulation
	// for simple input layer only "network_decoupled"
	string system_simu = "network_decoupled";
	
	// option for gradient computation
	// for simple input layer only "backprop_classic"
	string grad_comp = "backprop_classic";
	
	// gradient_check: compute numerical gradient (for dde_ibp) for comparision
	bool gradient_check = katana::getCmdOption_bool(argv, argv + argc, "-gradient_check", false);
	double epsilon_gradient_check = 1e-9;
	bool print_gradients_to_file = false;
	
	
	// ### ### ### --- PARAMETERS --- ### ### ### 
	
	// M = 784 and P = 10 are defined in global_constants.h
	
	
	// ... for the system/network architectur:
	
	int N = katana::getCmdOption(argv, argv + argc, "-N", 3072);  // number of nodes per hidden layer
	int L = katana::getCmdOption(argv, argv + argc, "-L", 5);  // number of hidden layers
	int D = katana::getCmdOption(argv, argv + argc, "-D", 50);  // number of delays (NOTE: must be the correct number also if diag_method = "full" or "from_file")
	
	double theta = katana::getCmdOption(argv, argv + argc, "-theta", 0.5);  // node separation
	double alpha = -1.0;  // factor in linear part of delay system
	
	string diag_method = katana::getCmdOption(argv, argv + argc, "-diag_method", "uniform");  // method to choose diagonals
	// "uniform": uniform distribution with the additional condition that there is at least one n'_d < 0 and one n'_d > 0
	// "equi_dist": diagonals with equal distances in between, if D is an odd number: main diagonal and same number of upper and lower, 
	//              if D is an even number and diag_distance is even: centered, e.g. -15, -5, 5, 15,
	//              if D is even and diag_distance is odd: centered around -0.5, e.g. -8, -3, 2, 7.
	// "from_file": diag indices are taken from a given text file
	// For a banded matrix as connection matrix choose equi_dist and diag_distance = 1
	// For a full connection matrix choose D = 2 * N - 1 (and either uniform or equi_dist with diag_distance = 1)
	int diag_distance = katana::getCmdOption(argv, argv + argc, "-diag_distance", 0);  // distance between diagonals if diag_method == "equi_dist",
							// minimum distance between diagonals if diag_method is "uniform", otherwise ignored
							// Do not choose diag_distance too large. If (D - 1) * (2 * diag_distance - 1) >= 2 * (N - diag_margin) - 1 and method is uniform, the program will abort.
	string diag_file_path = katana::getCmdOption(argv, argv + argc, "-diag_file_path", "diag.txt"); // path to text file containing D interger numbers n_prime_d, ignored if method is not "from_file" 
	
	
	// ... for the training:
	int number_of_epochs = katana::getCmdOption(argv, argv + argc, "-number_of_epochs", 100);
	double eta_0 = katana::getCmdOption(argv, argv + argc, "-eta0", 0.01);
	double eta_1 = katana::getCmdOption(argv, argv + argc, "-eta1", 1000.0);  // learning rate eta = min(eta_0, learning_rate_factor / step)
	bool pixel_shift = katana::getCmdOption_bool(argv, argv + argc, "-pixel_shift", false);  // on-off switch for training input random 1-pixel shift
	int max_pixel_shift = katana::getCmdOption(argv, argv + argc, "-max_pixel_shift", 1);
	bool input_noise = katana::getCmdOption_bool(argv, argv + argc, "-input_noise", false);  // on-off switch for training input gaussian noise
	double training_noise_sigma = katana::getCmdOption(argv, argv + argc, "-sigma", 0.01);  // standard deviation of gaussian noise to disturb training input
	bool rotation = katana::getCmdOption_bool(argv, argv + argc, "-rotation", false);
	double max_rotation_degrees = katana::getCmdOption(argv, argv + argc, "-max_rotation_degrees", 15.0);
	bool horizontal_flip = katana::getCmdOption_bool(argv, argv + argc, "-flip", false);  // only for CIFAR-10
	
	// ... for weight initialization:
	double initial_input_weigt_radius = sqrt(6.0 / ((double)D/2.0 + (double)M + 1.0));
	double initial_hidden_weigt_radius =  sqrt(6.0 / ((double)D + 1.0));
	double initial_output_weigt_radius =  sqrt(6.0 / ((double)D/2.0 + (double)P + 1.0));
	
	
	// ... for numerics:
	
	int N_h = max(32 ,(int)(16 * theta));  // computation steps per virtual node for solving DDE
	
	// computational precision for sums with exp(alpha * theta * n) factor in summands
	// in the functions "get_deltas" and "get_gradient" in "backprop.cpp". 
	// exp_precision = -35.0 means that terms, where exponential factor
	// is smaller than exp(-35), are ignored.
	// Since exp(-35) is approximately 6.3e-16, the gradient will still be computed with double precision. 
	double exp_precision = -35.0;
	
	
	
	// ### ### ### --- ETC. --- ### ### ###
	
	// make diag text file and end program if make_diags option is true
	if (make_diags){
		ofstream diag_file;
		diag_file.open("diag.txt");
		for (int i = 0; i < 6; ++i){
			vector<int> diag_indices = get_diag_indices(N, D, diag_method, diag_distance, diag_file_path, 0);
			for (int d = 0; d < D - 1; ++d){
				diag_file << diag_indices[d] << " ";
			}
			diag_file << diag_indices[D - 1] << endl;
		}
		diag_file.close();
		cout << "The option make_diags was true." << endl;
		cout << "Program made (or overrode) diag.txt" << endl;
		return 0;
	}
	
	// make initial weight files and diag file and end program if make_diags save_init_weights option is true
	if (save_init_weights){
		vector<int> diag_indices;
		ofstream diag_file;
		diag_file.open("diag.txt");
		for (int i = 0; i < 6; ++i){
			diag_indices = get_diag_indices(N, D, diag_method, diag_distance, diag_file_path, 0);
			for (int d = 0; d < D - 1; ++d){
				diag_file << diag_indices[d] << " ";
			}
			diag_file << diag_indices[D - 1] << endl;
		}
		diag_file.close();
		cout << "The option save_init_weights was true." << endl;
		cout << "Program made (or overrode) diag.txt" << endl;
		mat input_weights(N, M + 1);
		mat output_weights(P, N + 1);
		cube hidden_weights(L - 1, N, N + 1);
		initialize_weights(input_weights, hidden_weights, output_weights, D, L, N, diag_indices,
						   initial_input_weigt_radius, initial_hidden_weigt_radius, initial_output_weigt_radius, true);	
		cout << "Save initial weights and end program because the option save_init_weights is true." << endl;
		return 0;
	}
	
	// The following lines are to create a look up table "exp_table"
	// which contains the values exp(alpha * theta * n)
	// which are often needed by the functions "get_deltas" and "get_gradient" in "backprop.cpp". 
	vector<double> exp_table;
	int n = 0;
	while (alpha * theta * double(n) >= - 35.0){
		double exp_value = exp(alpha * theta * double(n));
		exp_table.push_back(exp_value);
		++n;
	}
	
	// make results file and print information about parameters
	print_parameters(results_file_name, print_msg, task,
					system_simu, grad_comp, gradient_check,
					N, L, D, theta, alpha,
					diag_method, diag_distance, diag_file_path,
					number_of_epochs, eta_0, eta_1,
					pixel_shift, input_noise, training_noise_sigma,
					N_h, exp_precision);
	
	// task "MNIST", "Fashion-MNIST", ...
	string data_dir;
	if (task == "MNIST"){
		data_dir = "data-MNIST";
	} else if (task == "Fashion-MNIST"){
		data_dir = "data-Fashion-MNIST";
	} else if (task == "CIFAR-10"){
		data_dir = "data-CIFAR-10";
	} else if (task == "SVHN"){
		data_dir = "data-SVHN";
	} else {
		cout << task << " is not a valid task." << endl;
		abort();
	}
	
	// read image data from files to arrays:
	cube train_images(number_of_training_batches, training_batch_size, M);
	mat test_images(test_batch_size, M);
	int train_labels[number_of_training_batches][training_batch_size];
	int test_labels[test_batch_size];
	read_files(train_images, test_images, train_labels, test_labels, data_dir);
	// The test images are not used at the moment.
	
	
	// for video:
	vector<int> step_indices_for_video;
	if (record_data){
		fstream index_file;
		string index_string;
		index_file.open("video/step_indices.txt", ios::in);
		do {
			getline(index_file, index_string);
			if (index_string != ""){
				int i = stoi(index_string);
				step_indices_for_video.push_back(i);
			}
		} while (index_string != "");
		index_file.close();
	}
	
	
	// ### ### ### --- TRAINING AND TEST --- ### ### ###
	

	
	
	
	vector<int> training_batch_indices;
	cout << "Begin training." << endl;
	for (int i = 0; i < number_of_training_batches; ++i){
		training_batch_indices.push_back(i);
	}




	// ### ### ### --- INITIALIZATION --- ### ### ###	

	// initialize arrays which are used below to store the current system states
	mat activations(L, N);
	mat node_states(L, N);
	double output_activations[P];
	double outputs[P];
	vec g_primes(N);

	// initialize arrays to store deltas and gradient
	mat deltas(L, N);
	double output_deltas[P];
	mat input_weight_gradient(N, M + 1, fill::zeros);
	mat output_weight_gradient(P, N + 1, fill::zeros);
	cube weight_gradient(L - 1, N, N + 1, fill::zeros);

	mat deltas_0(L, N);
	double output_deltas_0[P];
	mat input_weight_gradient_0(N, M + 1, fill::zeros);
	mat output_weight_gradient_0(P, N + 1, fill::zeros);
	cube weight_gradient_0(L - 1, N, N + 1, fill::zeros);

	// initialize arrays to eventually store numerical gradient
	mat num_input_weight_gradient(N, M + 1, fill::zeros);
	mat num_output_weight_gradient(P, N + 1, fill::zeros);
	cube num_weight_gradient(L - 1, N, N + 1, fill::zeros);


	// The function "get_diag_indices" returns D intergers n'_d between - N + 1 and N - 1.
	// The delays are then tau_d = N - n'_d.
	vector<int> diag_indices;
	diag_indices = get_diag_indices(N, D, diag_method, diag_distance, diag_file_path, 0);


	// initialize weights.
	mat input_weights(N, M + 1, fill::zeros);
	mat output_weights(P, N + 1, fill::zeros);
	cube hidden_weights(L - 1, N, N + 1, fill::zeros);
	if (init_weights_from_file){
		load_initial_weights(input_weights, hidden_weights, output_weights);
	} else {
		initialize_weights(input_weights, hidden_weights, output_weights, D, L, N, diag_indices,
					   initial_input_weigt_radius, initial_hidden_weigt_radius, initial_output_weigt_radius, false);
	}





	// ### ### ### --- STOCHASTIC GRADIENT DESCENT TRAINING --- ### ### ###


	// vectors to track training accuracy and validition accuracy (and eventually cosine similarity):
	vector<double> training_accuracy_vector;
	vector<double> accuracy_vector;
	vector<double> similarity_vector;

	// vector and variables to measure and save cpu time needed for each epoch 
	vector<double> time_vector;
	clock_t start;
	clock_t ende;
	double epoch_time;


	// loop over training epochs:
	for (int epoch = 0; epoch < number_of_epochs; ++epoch){

		start = clock();

		// make vector with randomly shuffled indices between 0 and 59999 (for MNIST, different number for SVHN) for each epoch of the stochastic gradient descent 
		vector<int> index_vector;
		for (int index = 0; index < number_of_training_batches*training_batch_size; ++index){
			index_vector.push_back(index);
		}
		shuffle(begin(index_vector), std::end(index_vector), rng);

		// loop over single training steps:
		int step_index = 0;
		for (int index : index_vector){
			++step_index;
			cout << step_index << " of " << number_of_training_batches*training_batch_size << endl;


			// record data for video: step 0 
			if (record_data && epoch == 0 && step_index == 1){
				int rec_step = 0;
				cout << "record data after step: " << rec_step << endl;
				//save weights
				string input_file_name = "video/weights_input_step_" + to_string(rec_step) + ".txt";
				string hidden_file_name = "video/weights_hidden_step_" + to_string(rec_step) + ".txt";
				string output_file_name = "video/weights_output_step_" + to_string(rec_step) + ".txt";
				input_weights.save(input_file_name, csv_ascii);
				hidden_weights.save(hidden_file_name, raw_ascii);
				output_weights.save(output_file_name, csv_ascii);
				//run system with examples from validation batch and save input, output and x_states
				for (int example_index = 0; example_index < number_of_examples; ++example_index){
					// select image as input
					vec input_data = test_images.row(example_index).t();
					int label = test_labels[index];
					// run system and record data for x_states
					solve_dde_ibp(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha,
							  N, L, N_h, example_index+1, rec_step);
					// save input and output
					string input_file_name = "video/vector_input_step_" + to_string(rec_step) + "_example_" + to_string(example_index + 1) + ".txt";
					input_data.save(input_file_name, csv_ascii);
					string output_file_name = "video/vector_output_step_" + to_string(rec_step) + "_example_" + to_string(example_index + 1) + ".txt";
					ofstream output_file;
					output_file.open(output_file_name);
					for (double y_p : outputs){
						output_file << y_p << endl;
					}
					output_file.close();
				}
			}


			double eta = learning_rate(epoch, step_index, eta_0, eta_1);

			// select image as input
			div_t div_result = div(index, training_batch_size);
			int batch_index = training_batch_indices[div_result.quot];
			int image_index = div_result.rem;
			vec input_data = train_images.tube(batch_index, image_index);
			int label = train_labels[batch_index][image_index];

			// data augmentation or regression
			if (rotation && M == 3072){
				double rotation_degrees = uniform(-max_rotation_degrees, max_rotation_degrees);
				rotation32(input_data, rotation_degrees);
			}

			if (pixel_shift && M == 784){
				pixel_shift28(input_data, max_pixel_shift);
			}
			if (pixel_shift && M == 3072){
				pixel_shift32(input_data, max_pixel_shift);
			}
			if (input_noise){
				vec training_noise = training_noise_sigma * vec(M, fill::randn);
				input_data += training_noise;
			}
			if (horizontal_flip && M == 3072){
				double random_val = uniform(0,1);
				if (random_val > 0.5){
					horizontal_flip32(input_data);
				}
			}

			// solve the DDE (or network)
			clock_t start_solve = clock();
			if (system_simu == "dde_ibp"){
				solve_dde_ibp(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha,
							  N, L, N_h);
			} else if (system_simu == "dde_heun"){
				solve_dde_heun(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha,
							  N, L, N_h);
			} else if (system_simu == "network"){
				solve_network(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, theta, alpha,
							  N, L);
			} else if (system_simu == "network_decoupled"){
				solve_network_decoupled_fast(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, N, L, diag_indices);
			} else {
				cout << system_simu << " is not a valid value for the system_simu option." << endl;
				abort();
			}
			clock_t end_solve = clock();
			cumulative_solve_time += (end_solve - start_solve) / (double)CLOCKS_PER_SEC;

			// get target vector
			double targets[P];
			get_targets(targets, label);

			// compute deltas and gradient
			clock_t start_backprop = clock();
			mat f_prime_activations = f_prime_matrix(activations);
			if (grad_comp == "backprop_standard"){
				get_deltas(deltas, output_deltas, outputs, targets, f_prime_activations, hidden_weights, output_weights, diag_indices, theta, alpha, N, L, exp_table);
				get_gradient(input_weight_gradient, weight_gradient, output_weight_gradient, deltas, output_deltas, input_data, node_states, f_prime_activations, g_primes, diag_indices, theta, alpha, N, L, exp_table);
			} else if (grad_comp == "backprop_classic"){
				get_gradient_classical_backprop(input_weight_gradient, weight_gradient, output_weight_gradient, input_data, node_states, f_prime_activations, g_primes, outputs, targets, hidden_weights, output_weights, diag_indices, N, L);
			} else {
				cout << grad_comp << " is not a valid value for the grad_comp option." << endl;
				abort();
			}
			clock_t end_backprop = clock();
			cumulative_backprop_time += (end_backprop - start_backprop) / (double)CLOCKS_PER_SEC;
			// eventually compute numerical gradient for comparision and save numerical gradient and backprop gradient in files
			if (gradient_check && step_index == 1){
				if (system_simu != "dde_ibp"){
					cout << "The option to compute the gradient numerically is only available if the method 'dde_ibp' is used to solve the delay system. " << endl;
					cout << "Set the option 'system_simu' to 'dde_ibp' or set the option 'num_gradient' to false." << endl;
					abort();
				}
				get_num_gradient(num_input_weight_gradient, num_weight_gradient, num_output_weight_gradient, 
								 activations, node_states, output_activations, outputs, g_primes, label,
								 input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha, N, L, N_h, epsilon_gradient_check);
				if (print_gradients_to_file){
					write_gradients_to_file(input_weight_gradient, weight_gradient, output_weight_gradient, num_input_weight_gradient, num_weight_gradient, num_output_weight_gradient, epoch + 1, step_index, 0);
				}
				double cos_sim = cosine_similarity(input_weight_gradient, weight_gradient, output_weight_gradient, num_input_weight_gradient, num_weight_gradient, num_output_weight_gradient);
				similarity_vector.push_back(cos_sim);
			}

			// compare gradient with old modified bp
			/*
			get_deltas(deltas_0, output_deltas_0, outputs, targets, f_prime_activations, hidden_weights, output_weights, diag_indices, theta, alpha, N, L, exp_table);
			get_gradient(input_weight_gradient_0, weight_gradient_0, output_weight_gradient_0, deltas_0, output_deltas_0, input_data, node_states, f_prime_activations, g_primes, diag_indices, theta, alpha, N, L, exp_table);
			double cos_sim = cosine_similarity(input_weight_gradient, weight_gradient, output_weight_gradient, input_weight_gradient_0, weight_gradient_0, output_weight_gradient_0);
			cout << cos_sim << endl;
			write_gradients_to_file(input_weight_gradient, weight_gradient, output_weight_gradient, input_weight_gradient_0, weight_gradient_0, output_weight_gradient_0, epoch + 1, step_index, validation_batch_index);
			*/

			weight_gradient = weight_gradient;
			output_weight_gradient = output_weight_gradient;

			// perform weight updates
			hidden_weights += - eta * weight_gradient;
			output_weights += - eta * output_weight_gradient;



			// record data for video: step 1 to end 
			if (record_data && find(step_indices_for_video.begin(), step_indices_for_video.end(), epoch * 50000 + step_index) != step_indices_for_video.end()){
				int rec_step = epoch * 50000 + step_index;
				cout << "record data after step: " << rec_step << endl;
				//save weights
				string input_file_name = "video/weights_input_step_" + to_string(rec_step) + ".txt";
				string hidden_file_name = "video/weights_hidden_step_" + to_string(rec_step) + ".txt";
				string output_file_name = "video/weights_output_step_" + to_string(rec_step) + ".txt";
				input_weights.save(input_file_name, csv_ascii);
				hidden_weights.save(hidden_file_name, raw_ascii);
				output_weights.save(output_file_name, csv_ascii);
				//run system with examples from validation batch and save input, output and x_states
				for (int example_index = 0; example_index < number_of_examples; ++example_index){
					// select image as input
					vec input_data = test_images.row(example_index).t();
					int label = test_labels[index];
					// run system and record data for x_states
					solve_dde_ibp(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha,
							  N, L, N_h, example_index+1, rec_step);
					// save input and output
					string input_file_name = "video/vector_input_step_" + to_string(rec_step) + "_example_" + to_string(example_index + 1) + ".txt";
					input_data.save(input_file_name, csv_ascii);
					string output_file_name = "video/vector_output_step_" + to_string(rec_step) + "_example_" + to_string(example_index + 1) + ".txt";
					ofstream output_file;
					output_file.open(output_file_name);
					for (double y_p : outputs){
						output_file << y_p << endl;
					}
					output_file.close();
				}
			}
		}


		// loop to get accuracy on training set:
		int correct_count = 0;  // counter for calculating accuracy
		for (int index = 0; index < number_of_training_batches * training_batch_size; ++index){

			// select image as input
			div_t div_result = div(index, training_batch_size);
			int batch_index = training_batch_indices[div_result.quot];
			int image_index = div_result.rem;
			vec input_data = train_images.tube(batch_index, image_index);
			int label = train_labels[batch_index][image_index];

			// solve the DDE (or network)
			clock_t start_solve = clock();
			if (system_simu == "dde_ibp"){
				solve_dde_ibp(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha,
							  N, L, N_h);
			} else if (system_simu == "dde_heun"){
				solve_dde_heun(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha,
							  N, L, N_h);
			} else if (system_simu == "network"){
				solve_network(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, theta, alpha,
							  N, L);
			} else if (system_simu == "network_decoupled"){
				solve_network_decoupled_fast(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, N, L, diag_indices);
			} else {
				cout << system_simu << " is not a valid value for the system_simu parameter." << endl;
				abort();
			}
			clock_t end_solve = clock();
			cumulative_solve_time += (end_solve - start_solve) / (double)CLOCKS_PER_SEC;

			// get target vector
			double targets[P];
			get_targets(targets, label);

			int output_result = distance(outputs, max_element(outputs, outputs + P));
			if (label == output_result){
				++correct_count;
			}
		}
		training_accuracy_vector.push_back(double(correct_count) / (double)(number_of_training_batches*training_batch_size/100.0));


		// loop for validation:
		correct_count = 0;
		for (int index = 0; index < test_batch_size; ++index){

			vec input_data = test_images.row(index).t();
			int label = test_labels[index];

			// solve the DDE (or network)
			clock_t start_solve = clock();
			if (system_simu == "dde_ibp"){
				solve_dde_ibp(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha,
							  N, L, N_h);
			} else if (system_simu == "dde_heun"){
				solve_dde_heun(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, diag_indices, theta, alpha,
							  N, L, N_h);
			} else if (system_simu == "network"){
				solve_network(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, theta, alpha,
							  N, L);
			} else if (system_simu == "network_decoupled"){
				solve_network_decoupled_fast(activations, node_states, output_activations, outputs, g_primes,
							  input_data, input_weights, hidden_weights, output_weights, N, L, diag_indices);
			} else {
				cout << system_simu << " is not a valid value for the system_simu parameter." << endl;
				abort();
			}
			clock_t end_solve = clock();
			cumulative_solve_time += (end_solve - start_solve) / (double)CLOCKS_PER_SEC;

			// get target vector
			double targets[P];
			get_targets(targets, label);

			int output_result = distance(outputs, max_element(outputs, outputs + P));
			if (label == output_result){
				++correct_count;
			}
		}
		accuracy_vector.push_back(double(correct_count) / (double)(test_batch_size/100.0));

		cout << "epoch " << epoch + 1 << ": validation accuracy = " << double(correct_count) / (double)(test_batch_size/100.0) << endl;

		// eventually print weights to file at end of each epoch
		if (print_weights_to_file){
			print_weights(input_weights, hidden_weights, output_weights, diag_indices, 0, epoch);
		}

		ende = clock();
		epoch_time = ((double) (ende - start)) / (double)CLOCKS_PER_SEC;
		time_vector.push_back(epoch_time);
	}

	// print result to file:
	print_results(results_file_name, diag_indices, training_accuracy_vector, accuracy_vector, similarity_vector);

	
	// for cpu time measuring
	clock_t end_overall = clock();
	double cpu_time_overall = (end_overall - start_overall) / (double)CLOCKS_PER_SEC;
	double cpu_time_residual = cpu_time_overall - cumulative_solve_time - cumulative_backprop_time;
	double cpu_time_solve_percentage = 100.0 * cumulative_solve_time / cpu_time_overall;
	double cpu_time_backprop_percentage = 100.0 * cumulative_backprop_time / cpu_time_overall;
	double cpu_time_residual_percentage = 100.0 * cpu_time_residual / cpu_time_overall;
	
	// get current time and date and print to results text file
	auto current_clock = chrono::system_clock::now();
	time_t current_time = chrono::system_clock::to_time_t(current_clock);
	ofstream results_file;
	results_file.open(results_file_name, ios_base::app);
	results_file << endl;
	results_file << endl;
	results_file << "total cpu time (in seconds): " << cpu_time_overall << endl;
	results_file << "cumulative cpu time for solving the DDE or network (in seconds): " << cumulative_solve_time << " (" << cpu_time_solve_percentage << "%)" << endl;
	results_file << "cumulative cpu time for backpropagation (in seconds): " << cumulative_backprop_time << " (" << cpu_time_backprop_percentage << "%)" << endl;
	results_file << "residual cpu time (in seconds): " << cpu_time_residual << " (" << cpu_time_residual_percentage << "%)" << endl;
	results_file << endl;
	results_file << "end of simulation: " << ctime(&current_time);
	results_file.close();
	
	
	return 0;
}