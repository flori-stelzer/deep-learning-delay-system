# include "helper_functions.h"

using namespace std;
using namespace arma;
using namespace globalconstants;


auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
mt19937 rng(seed);


double uniform(double a, double b){
	/*
	function returns random value from uniform distribution on [a, b]
	*/
	uniform_real_distribution<double> dis(a, b);
	return dis(rng);
}


int uniform_int(int a, int b){
	/*
	function returns random integer value from discrete uniform distribution on {a, ...., b - 1}.
	*/
	uniform_int_distribution<int> dis(a, b - 1);
	return dis(rng);
}



double learning_rate(int epoch, int k, double eta_0, double eta_1){
	/*
	Function to compute the learning rate eta in dependence of the step k (and epoch).
	*/
	double k_total = epoch * (number_of_training_batches - 1)*training_batch_size + k;
	return min (eta_0, eta_1 / k_total);
}


void read_files(cube &train_images, mat &test_images, int (&train_labels)[number_of_training_batches][training_batch_size],int (&test_labels)[test_batch_size], string data_dir){
	/*
	Function to open the txt files in the data folder.
	Reads the images stored in these files to an armadillo cube resp. matrix
	and the labels to int arrays.
	
	Args:
	train_images: reference to arma::cube with 6 rows, 10000 cols, 784 slices (or corresponding numbers if task != MNIST).
	              To be filled with training image data.
				  Each row represents one training data batch,
				  each col of represents one image,
				  each slice represents one pixel.
	test_images:  reference to arma::matrix with 10000 rows, 784 cols (or corresponding numbers if task != MNIST).
	              To be filled with test image data.
				  Each row of represents one image,
				  each col (of a row) represents one pixel.
	train_labels: reference to an int array with size number_of_training_batches x training_batch_size (e.g. 6 x 10 for MNIST).
	              To be filled with labels (0, 1, ..., 9) of the training images.
	test_labels:  reference to an int array with size test_batch_size (e.g. 10000 for MNIST).
	              To be filled with labels of the test images.
	data_dir:     string
	              "../" + data_dir is the path to the directory containing the MNIST resp. Fashion-MNIST data set.
	*/
	
	//open training images and store data in armadillo cube normalized to [0, 1]
	//row is batch, col is image, slice is pixel
	fstream train_images_file;
	string train_image_string;
	for(int batch_index = 0; batch_index < number_of_training_batches; ++batch_index){
		train_images_file.open("../" + data_dir + "/train_images_" + to_string(batch_index + 1) + ".txt", ios::in);
		for(int image_index = 0; image_index < training_batch_size; ++image_index){
			getline(train_images_file, train_image_string);
			for(int pixel_index = 0; pixel_index < M; ++pixel_index){
				string hex_code = train_image_string.substr(2 * pixel_index, 2);
				int int_pixel;
				stringstream s;
				s << hex << hex_code;
				s >> int_pixel;
				double double_pixel = int_pixel;
				train_images(batch_index, image_index, pixel_index) = double_pixel/255;
			}
		}
		train_images_file.close();
	}
	
	//open test images file and store data in armadillo matrix normalized to [0, 1]
	//row is image, col is pixel
	fstream test_images_file;
	string test_image_string;
	test_images_file.open("../" + data_dir + "/test_images.txt", ios::in);
	for(int row = 0; row < test_batch_size; ++row){
		getline(test_images_file, test_image_string);
		for(int col = 0; col < M; ++col){
			string hex_code = test_image_string.substr(2 * col, 2);
			int int_pixel;
			stringstream s;
			s << hex << hex_code;
			s >> int_pixel;
			double double_pixel = int_pixel;
			test_images(row, col) = double_pixel/255;
		}
	}
	test_images_file.close();
	
	//open training label files and store labels in int array
	for(int batch_index = 0; batch_index < number_of_training_batches; ++batch_index){
		fstream train_labels_file;
		train_labels_file.open("../" + data_dir + "/train_labels_" + to_string(batch_index + 1) + ".txt", ios::in);
		string train_labels_string;
		getline(train_labels_file, train_labels_string);
		train_labels_file.close();
		if (data_dir == "data-CIFAR-100-coarse"){
			for(int image_index = 0; image_index < training_batch_size; ++image_index){
				char c1 = train_labels_string[2*image_index];
				char c2 = train_labels_string[2*image_index+1];
				train_labels[batch_index][image_index] = 10 * (c1 - '0') + (c2 - '0');
			}
		} else {
			for(int image_index = 0; image_index < training_batch_size; ++image_index){
				char c = train_labels_string[image_index];
				train_labels[batch_index][image_index] = c - '0';
			}
		}
	}
	
	//open test label file and store labels in int array
	fstream test_labels_file;
	test_labels_file.open("../" + data_dir + "/test_labels.txt", ios::in);
	string test_labels_string;
	getline(test_labels_file, test_labels_string);
	test_labels_file.close();
	if (data_dir == "data-CIFAR-100-coarse"){
		for(int image_index = 0; image_index < test_batch_size; ++image_index){
			char c1 = test_labels_string[2*image_index];
			char c2 = test_labels_string[2*image_index+1];
			test_labels[image_index] = 10 * (c1 - '0') + (c2 - '0');
		}
	} else {
		for(int image_index = 0; image_index < test_batch_size; ++image_index){
			char c = test_labels_string[image_index];
			test_labels[image_index] = c - '0';
		}
	}
}


void initialize_weights(mat &input_weights, cube &hidden_weights, mat &output_weights, int D, int L, int N, vector<int> diag_indices, 
						double initial_input_weigt_radius, double initial_hidden_weigt_radius, double initial_output_weigt_radius, bool save_to_file, bool no_input_layer){
	/*
	Function to initialize weigths.
	
	Args:
	input_weights:      reference to arma::mat of size N x (M + 1)
	                    Matrix W^in. To be filled with the initial weights connecting the input layer
				        to the first hidden layer (including the input bias weight).
	hidden_weights:     reference to arma::cube with L - 1 rows, N cols, N + 1 slices.
	                    To be filled with the initial weights between the hidden layers.
						1st index: layer (where the connection ends),
						2nd index: node index of layer where the connection ends,
						3rd index: node index of layer where the connections begins.
	output_weights:     reference to arma::mat with size P x (N + 1) (where P = 10).
	                    To be filled with the initial weights connecting the last hidden layer to the output layer.
	D:                  int.
						Number of delays = number of diagonals with nonzero entries in hidden weight matrices.
	L:                  int.
						Number of hidden layers.
	N:                  int.
						Nodes per hidden layer (except bias node).
	diag_indices:       int vector of length D.
	                    Contains the indices n'_d for the nonzero diagonals of the hidden weight matrices.
	initial_input_weigt_radius:  double.
	initial_hidden_weigt_radius: double.
	initial_output_weigt_radius: double.
	save_to_file:       bool
						if true: save initial weights to file and abort program.
	*/
	
	// initial input weights
	if(no_input_layer)
	{
		input_weights.eye(N, M + 1);  //set the input to an identity matrix, implementing a purely sequential input
	}
	else
	{
		input_weights = - initial_input_weigt_radius * mat(N, M + 1, fill::ones) + 2.0 * initial_input_weigt_radius * mat(N, M + 1, fill::randu);
	}
	
	// initial output weights
	output_weights = - initial_output_weigt_radius * mat(P, N + 1, fill::ones) + 2.0 * initial_output_weigt_radius * mat(P, N + 1, fill::randu);
	
	// hidden weights, nonzero only on certain diagonals:
	for (int l = 0; l < L - 1; ++l){
		for (int diag_index : diag_indices){
			int n_min = max(0, diag_index);
			int n_max = min(N, N + diag_index);
			for (int n = n_min; n < n_max; ++n){
				int j = n - diag_index;
				hidden_weights(l, n, j) =  uniform(-initial_output_weigt_radius, initial_output_weigt_radius);
			}
		}
		for (int n = 0; n < N; ++n){
			hidden_weights(l, n, N) = uniform(-initial_output_weigt_radius, initial_output_weigt_radius);  // initial bias weight (for on-neuron)
		}
	}
	
	// eventually save initial weights
	if (save_to_file){
		input_weights.save("weights/init_input_weights");
		output_weights.save("weights/init_output_weights");
		hidden_weights.save("weights/init_hidden_weights");
	}
}


void load_initial_weights(mat &input_weights, cube &hidden_weights, mat &output_weights){
	/*
	Function to initialize weigths.
	
	Args:
	input_weights:      reference to arma::mat of size N x (M + 1)
	                    Matrix W^in. To be filled with the initial weights connecting the input layer
				        to the first hidden layer (including the input bias weight).
	hidden_weights:     reference to arma::cube with L - 1 rows, N cols, N + 1 slices.
	                    To be filled with the initial weights between the hidden layers.
						1st index: layer (where the connection ends),
						2nd index: node index of layer where the connection ends,
						3rd index: node index of layer where the connections begins.
	output_weights:     reference to arma::mat with size P x (N + 1) (where P = 10).
	                    To be filled with the initial weights connecting the last hidden layer to the output layer.
	*/
	input_weights.load("weights/init_input_weights");
	output_weights.load("weights/init_output_weights");
	hidden_weights.load("weights/init_hidden_weights");
}



vector<int> get_diag_indices(int N, int D, string method, int diag_distance, string diag_file_path, int line_index){
	/*
	Function to choose D diagonal indices n' from set {-N + 1, ..., N - 1}
	
	One of the following methods can be chosen to do so:
	uniform:	Each n'_d is drawn from a discrete uniform distribution on {-N + 1, ..., N - 1}.
				If a drawn n'_d was already chosen before or if distance too small: drop and repeat.
				Moreover, there must be at least one positive and one negative index.
				Minimum distance between diagonals given by diag_distance.
	equi_dist:	The diagonal indices n'_d are deterministic and equally spaced.
				Case D is odd:	The main diagonal is always chosen, i.e. one of the n'_d is the number 0.
								Minimun index is n'_1 = - diag_distance * (D - 1) / 2,
								i.e., there are (D - 1)/2 negative n'_d and (D - 1)/2 positive n'_d.
				Case D is even:	If diag_distance is even too: min index is n'_d = - diag_distance / 2 - (D / 2 - 1) * diag_distance,
								i.e. diag indices are centered around 0.
								If diag_distance is odd: n'_1 = - (diag_distance + 1) / 2 - (D / 2 - 1) * diag_distance,
								i.e. diag indices are centered around -0.5.
	from_file:	The number n'_d are taken from a given text file (from a specified line).
	
	Args:
	N:				int.
					Number of nodes per hidden layer.
	D:				int.
					Number of nonzero diagonals.
	method:			string.
					Method to choose the nonzero diagonal indices.
	diag_distance:	int.
					If method is equi_dist: distance between nonzero diagonals.
					If method is uniform: minimum distance between nonzero diagonals.
	diag_file_path:	string.
					If method is from_file: path to text file containing the diagonal indices.
	line_index:		int.
					If method is from_file: line in text file from which the diagonal indices should be taken.	
	*/
		
	if (2 * N - 1 < D){
		cout << "Not enough space for D diagonals." << endl;
		cout << "N = " << N << ", D = " << D << endl;
		abort();
	}
	
	vector<int> diag_indices;
	
	if (method == "uniform"){
		// draw n_prime_d values from uniform distribution.
		// drop value if already in diag_indices vector.
		// repeat until D n_primes were chosen.
		if ((D - 1) * (2 * diag_distance - 1) >= 2 * N - 1){
			cout << "N = " << N << ", D = " << D << " and diag_distance = " << diag_distance << endl;
			cout << "The condition (D - 1) * (2 * diag_distance - 1) < 2 * N - 1 for the method uniform is violated" << endl;
			abort();
		}
		do{
			diag_indices.clear();
			while (diag_indices.size() < (unsigned long) D){
				int random_value = uniform_int(1 - N, N);
				// check whether distance is at least diag_distance
				bool add_number = true;
				for (int n_prime : diag_indices){
					if (abs(random_value - n_prime) < diag_distance){
						add_number = false;
						break;
					}
				}
				// add random_value to diag_indices
				if (add_number && find(diag_indices.begin(), diag_indices.end(), random_value) == diag_indices.end()){
					diag_indices.push_back(random_value);
				}
			}
			sort(diag_indices.begin(), diag_indices.end());
		} while (diag_indices[0] >= 0 || diag_indices[D - 1] <= 0);  // check is there is at least one positive and one negative index
	} else if (method == "equi_dist"){
		int min_n_prime;
		if (D % 2 == 0){
			if (diag_distance % 2 == 0){
				min_n_prime = - (D / 2 - 1) * diag_distance - diag_distance / 2;
			} else {
				min_n_prime = - (D / 2 - 1) * diag_distance - (diag_distance + 1) / 2;
			}
		} else {
			min_n_prime = - ((D - 1) / 2) * diag_distance;
		}
		if (min_n_prime < 1 - N){
			cout << "D = " << D << ", diag_distance = " << diag_distance << " and N = " << N << " not possible." << endl;
			abort();
		}
		for (int d = 0; d < D; ++d){
			int n_prime_d = min_n_prime + d * diag_distance;
			diag_indices.push_back(n_prime_d);
		}
	} else if (method == "from_file"){
		// read file given by diag_file_path and take diag_indices from (line_index + 1)-th line
		string diag_string;
		fstream diag_file;
		diag_file.open(diag_file_path, ios::in);
		for (int i = 0; i < line_index + 1; ++i){
			getline(diag_file, diag_string);
		}
		stringstream diag_stream(diag_string);
		int n_prime;
		while (diag_stream >> n_prime){
			diag_indices.push_back(n_prime);
		}
		sort(diag_indices.begin(), diag_indices.end());
	} else {
		cout << method << " is not a valid method for the get_diag function." << endl;
		abort();
	}
	
	// check if diag_indices has the correct size (could be wrong if method is "full" or "from_file" and D is not chosen properly.)
	if (diag_indices.size() != (unsigned long) D){
		cout << "D = " << D << ", but diag_indices.size() = " << diag_indices.size() << endl;
		abort();
	}
	
	return diag_indices;
}


void get_targets(double (&targets)[P], int label){
	/*
	Function to convert label (e.g. 2) to target vector (e.g. (0, 0, 1, 0, ...)).
	
	Args:
	targets: reference to double array of length P = 10.
	         To be filled with 0.0 and 1.0, where the position of the 1.0 is determined by the label.
	label: int.
	       Number between 0 and 9.
	*/
	for (int p = 0; p < P; ++p){
		if (p == label){
			targets[p] = 1.0;
		} else {
			targets[p] = 0.0;
		}
	}
}


void pixel_shift28(arma::vec &input_data, int max_pixel_shift){
	/*
	Function to shift the input image randomly by at most max_pixel_shift pixel per direction.
	I.e. if max_pixel_shift==1, then there are 9 possible variants.
	The image is always shifted by a full pixel, so no interpolation is necessary.
	The resulting empty pixels at the margin of the image are filled with zeros (white).
	
	Args:
	input_data:		 reference to arma::vec of length M = 784.
					 Input vector. Contains the pixel values of an input image.
	max_pixel_shift: int
					 Maximum pixel shift distance.
	*/
	
	// reshape input data to 2D image
	mat image_original(28, 28);
	for (int i = 0; i < M; ++i){
		int x = i / 28;
		int y = i % 28;
		image_original(x, y) = input_data(i);
	}
	
	// select random translation distances
	int vert_shift = uniform_int(-max_pixel_shift, max_pixel_shift+1);
	int hori_shift = uniform_int(-max_pixel_shift, max_pixel_shift+1);
	
	// shift image
	mat image_new(28, 28, fill::zeros);
	int i_min = max(0, vert_shift);
	int i_max = min(28, 28 + vert_shift);
	int j_min = max(0, hori_shift);
	int j_max = min(28, 28 + hori_shift);
	for (int i = i_min; i < i_max; ++i){
		for (int j = j_min; j < j_max; ++j){
			image_new(i, j) = image_original(i - vert_shift, j - hori_shift);
		}
	}
	
	// reshape new image to vector
	for (int x = 0; x < 28; ++x){
		for (int y = 0; y < 28; ++y){
			int i = 28 * x + y;
			input_data(i) = image_new(x, y);
		}
	}
}


void pixel_shift32(arma::vec &input_data, int max_pixel_shift){
	/*
	Same as pixel_shift28 but for SVNH or CIFAR 23x23 rgb-images which have the following format:
	(r11, g11, b11, r12, g12, b12, ...)
	*/
	
	// select random shift distances -1, 0 or 1
	int vert_shift = uniform_int(-max_pixel_shift, max_pixel_shift+1);
	int hori_shift = uniform_int(-max_pixel_shift, max_pixel_shift+1);
	
	// horizontal shift:
	while (hori_shift > 0){
		for (int row = 0; row < 32; ++row){
			for (int col = 31; col > 0; --col){
				input_data(3*32*row + 3*col) = input_data(3*32*row + 3*(col-1));
				input_data(3*32*row + 3*col + 1) = input_data(3*32*row + 3*(col-1) + 1);
				input_data(3*32*row + 3*col + 2) = input_data(3*32*row + 3*(col-1) + 2);
			}
		}
		--hori_shift;
	}
	while (hori_shift < 0){
		for (int row = 0; row < 32; ++row){
			for (int col = 0; col < 31; ++col){
				input_data(3*32*row + 3*col) = input_data(3*32*row + 3*(col+1));
				input_data(3*32*row + 3*col + 1) = input_data(3*32*row + 3*(col+1) + 1);
				input_data(3*32*row + 3*col + 2) = input_data(3*32*row + 3*(col+1) + 2);
			}
		}
		++hori_shift;
	}
	
	// vetical shift:
	while (vert_shift > 0){
		for (int row = 31; row > 0; --row){
			for (int col = 0; col < 32; ++col){
				input_data(3*32*row + 3*col) = input_data(3*32*(row-1) + 3*col);
				input_data(3*32*row + 3*col + 1) = input_data(3*32*(row-1) + 3*col + 1);
				input_data(3*32*row + 3*col + 2) = input_data(3*32*(row-1) + 3*col + 2);
			}
		}
		--vert_shift;
	}
	while (vert_shift < 0){
		for (int row = 0; row < 31; ++row){
			for (int col = 0; col < 32; ++col){
				input_data(3*32*row + 3*col) = input_data(3*32*(row+1) + 3*col);
				input_data(3*32*row + 3*col + 1) = input_data(3*32*(row+1) + 3*col + 1);
				input_data(3*32*row + 3*col + 2) = input_data(3*32*(row+1) + 3*col + 2);
			}
		}
		++vert_shift;
	}
}


void bilinear_interpolation32(vec &input_data, cube vectorfield){
	vec original_input_data = input_data;
	for (int i = 0; i < 32; ++i){
		for (int j = 1; j < 32; ++j){
			// y: row, x: column
			double x = vectorfield(i, j, 1);
			double y = vectorfield(i, j, 0);
			int x_floor = (int)floor(x);
			int y_floor = (int)floor(y);
			int x_ceil = (int)ceil(x);
			int y_ceil = (int)ceil(y);
			double x_frac = x - floor(x);
			double y_frac = y - floor(y);
			
			for (int color_index = 0; color_index < 3; ++color_index){
				int m = 3 * 32 * i + 3 * j + color_index;
				// m1: (x_floor, y_floor), m2: (x_ceil, y_floor),
				// m3: (x_floor, y_ceil), m4: (x_ceil, y_ceil), 
				int m1 = 3 * 32 * y_floor + 3 * x_floor + color_index;
				int m2 = 3 * 32 * y_floor + 3 * x_ceil + color_index;
				int m3 = 3 * 32 * y_ceil + 3 * x_floor + color_index;
				int m4 = 3 * 32 * y_ceil + 3 * x_ceil + color_index;
				input_data(m) = (1.0 - y_frac) * ((1.0 - x_frac) * original_input_data(m1) + x_frac * original_input_data(m2)) + y_frac * ((1.0 - x_frac) * original_input_data(m3) + x_frac * original_input_data(m4));
			}
		}
	}
}

void rotation32(vec &input_data, double rotation_degrees){
	/*
	positive degrees: counterclockwise rotation
	*/
	
	double rotation_angle = rotation_degrees * M_PI / 180.0;
	// make vectorfield
	cube vectorfield(32, 32, 2);
	for (int i = 0; i < 32; ++i){
		for (int j = 0; j < 32; ++j){
			// i: row, j: column
			double distance_to_center = sqrt((15.5 - double(i)) * (15.5 - double(i)) + (15.5 - double(j)) * (15.5 - double(j)));
			double pixel_angle;
			if (i < 16){
				pixel_angle = acos((double(j) - 15.5) / distance_to_center);
			} else {
				pixel_angle = 2.0 * M_PI - acos((double(j) - 15.5) / distance_to_center);
			}
			// y: row, x: column
			double x = cos(pixel_angle - rotation_angle) * distance_to_center + 15.5;
			double y = 15.5 - sin(pixel_angle - rotation_angle) * distance_to_center;
			// truncate if (x,y) is outside the frame
			if (x < 0.0){
				x = 0.0;
			}
			if (x > 31.0){
				x = 31.0;
			}
			if (y < 0.0){
				y = 0.0;
			}
			if (y > 31.0){
				y = 31.0;
			}
			vectorfield(i, j, 1) = x;
			vectorfield(i, j, 0) = y;
		}
	}
	
	// apply interpolation
	bilinear_interpolation32(input_data, vectorfield);
}

void horizontal_flip32(vec &input_data){
	double temp;
	for (int i = 0; i < 32; ++i){
		for (int j = 0; j < 16; ++j){
			for (int color_index = 0; color_index < 3; ++color_index){
				int m_left = 3 * 32 * i + 3 * j + color_index;
				int m_right = 3 * 32 * i + 3 * (31 - j) + color_index;
				temp = input_data(m_right);
				input_data(m_right) = input_data(m_left);
				input_data(m_left) = temp;
			}
		}
	}
}


void print_parameters(string results_file_name, string print_msg, string task, bool cross_validation, string system_simu, string grad_comp, bool gradient_check,
					int N, int L, int D, double theta, double alpha,
					string diag_method, int diag_distance, string diag_file_path,
					int number_of_epochs, double eta_0, double eta_1,
					bool pixel_shift,
					bool input_noise, double training_noise_sigma,
					int N_h, double exp_precision){
	/*
	Function to create text file in which the parameters and results will be saved.
	If a file with the same name already exists, it will be overridden.
	This functions prints the parameters and an initial message to the text file.
	The results will be appended later by the function "print_results".
	
	Args:
	results_file_name:	string.
						Name of the text file.
	print_msg:			string.
						Message to write at the beginning of the text file.
	all other:			diverse datatypes.
						Simulation options and parameters to be printed to the text file.
	*/
	
	// get current time and date
	auto current_clock = chrono::system_clock::now();
	time_t current_time = chrono::system_clock::to_time_t(current_clock);
	
	ofstream results_file;
	results_file.open(results_file_name);
	results_file << print_msg << endl;
	results_file << endl;
	results_file << "start of simulation: " << ctime(&current_time);
	results_file << endl;
	results_file << endl;
	results_file << "OPTIONS:" <<endl;
	results_file << endl;
	results_file << "task: " << task << endl;
	#ifdef TESTING
		results_file << "Training on all batches, testing on test data set." << endl;
	#else //Validation
		if (cross_validation){
			results_file << "6-fold cross_validation" << endl;
		} else {
			results_file << "no cross_validation, validation only on one batch" << endl;
		}
	#endif  //TESTING
	results_file << "method to solve the DDE (or network): " << system_simu << endl;
	results_file << "method to compute the gradient: " << grad_comp << endl;
	if (gradient_check){
		results_file << "gradient check for first training example of each epoch" << endl;
	} else {
		results_file << "gradient check: off" << endl;
	}
	results_file << endl;
	results_file << endl;
	results_file << "PARAMETERS:" <<endl;
	results_file << endl;
	results_file << "System Parameters:" << endl;
	results_file << "N = " << N << endl;
	results_file << "L = " << L << endl;
	results_file << "D = " << D << endl;
	results_file << "theta = " << theta << endl;
	results_file << "alpha = " << alpha << endl;
	results_file << "diag_method = " << diag_method << endl;
	results_file << "diag_distance = " << diag_distance << " (not relevant if diag_method is from_file)" << endl;
	results_file << "diag_file_path = " << diag_file_path << " (only relevant if diag_method is from_file)" << endl;
	results_file << endl;
	results_file << "Training Parameters:" << endl;
	results_file << "number_of_epochs = " << number_of_epochs << endl;
	results_file << "eta_0 = " << eta_0 << endl;
	results_file << "eta_1 = " << eta_1 << endl;
	if (pixel_shift){
		results_file << "random pixel shift: on" << endl;
	} else {
		results_file << "random pixel shift: off" << endl;
	}
	if (input_noise){
		results_file << "gaussian noise: on, sigma = " << training_noise_sigma << endl;
	} else {
		results_file << "gaussian noise: off" << endl;
	}
	results_file << endl;
	results_file << "Parameters for Numerics:" << endl;
	results_file << "numerical h-steps per virtual node = " << N_h << endl;
	results_file << "precision for exponential factors for delay-network-backpropagation = exp(" << exp_precision << ")" << endl;
	results_file << endl;
	results_file << endl;
	results_file << "RESULTS:" << endl;
	results_file.close();
}


void print_results(string results_file_name, int validation_batch_index, vector<int> diag_indices, vector<double> training_accuracy_vector, vector<double> accuracy_vector, vector<double> similarity_vector){
	/*
	Function to append results (i.e. training and validation accuracies for each epoch) to the results text file.
	Moreover the index of the validation batch and the diagonal indices n'_d are printed to the file.
	
	Args:
	results_file_name:			string.
								Name of the text file.
	validation_batch_index:		int.
								Index of the (currently) used batch for validation.
	diag_indices:				vector<int>.
								Vector containing the diagonal indices n'_d.
	training_accuracy_vector:	vector<double>.
								Vector containing the training accuracies for each epoch.
	accuracy_vector:			vector<double>.
								Vector containing the validation accuracies for each epoch.
	similarity_vector			vector<double>.
								Vector containing the cosine similarities between backprop gradient
								and numerically computed gradient if gradient check is switched on,
								otherwise empty vector.
	*/
	int epochs = accuracy_vector.size();
	
	ofstream results_file;
	results_file.open(results_file_name, ios_base::app);
	results_file << endl;
#ifdef TESTING
	results_file << "Results for test set" << endl;
#else //Validation
	results_file << "Results for Validation Batch " << validation_batch_index + 1 << endl;
#endif
	results_file << "diagonal indices n_prime_d: ";
	for (uint i = 0; i < diag_indices.size() - 1; ++i){
		results_file << diag_indices[i] << ", ";
	}
	results_file << diag_indices[diag_indices.size() - 1] << endl;
#ifdef TESTING
	results_file << "epoch_number " << "training_accuracy " << "test_accuracy";
#else
	results_file << "epoch_number " << "training_accuracy " << "validation_accuracy";
#endif
	if (similarity_vector.empty()){
			results_file << endl;
		} else {
		results_file << " cosine_similarity" << endl;
	}
	for (int e = 0; e < epochs; ++e){
		stringstream stream_train;
		stream_train << fixed << setprecision(6) << training_accuracy_vector[e];
		string s_train = stream_train.str();
		stringstream stream_val;
		stream_val << fixed << setprecision(6) << accuracy_vector[e];
		string s_val = stream_val.str();
		string cos_sim;
		if (similarity_vector.empty()){
			cos_sim = "";
		} else {
			stringstream stream_sim;
			stream_sim << fixed << setprecision(12) << similarity_vector[e];
			cos_sim = " " + stream_sim.str();
		}
		if (e + 1 < 10){
			results_file << "00" << e + 1 << " " << s_train << " " << s_val << cos_sim << endl;
		} else if (e + 1 < 100){
			results_file << "0" << e + 1 << " " << s_train << " " << s_val << cos_sim << endl;
		} else {
			results_file << e + 1 << " " << s_train << " " << s_val << cos_sim << endl;
		}
	}
	results_file.close();
}


void print_weights(mat input_weights, cube hidden_weights, mat output_weights, vector<int> diag_indices, int validation_batch_index, int epoch){
	/*
	Function to print the weights (and diagonal indices) to text files in the "weights" directory.
	For identification the text files have the following file names:
		"diag_indices_val_batch_${validation_batch}_epoch_${epoch}.txt"
		"input_weights_val_batch_${validation_batch}_epoch_${epoch}.txt"
		"hidden_weights_val_batch_${validation_batch}_epoch_${epoch}.txt"
		"output_weights_val_batch_${validation_batch}_epoch_${epoch}.txt"
	
	Args:
	input_weights:			arma::mat of size N x (M + 1)
							Matrix W^in. Filled with the weights connecting the input layer
							to the first hidden layer (including the input bias weight).
	hidden_weights:			arma::cube with L - 1 rows, N cols, N + 1 slices.
							Filled with the weights between the hidden layers.
							1st index: layer (where the connection ends),
							2nd index: node index of layer where the connection ends,
							3rd index: node index of layer where the connections begins.
	output_weights:			arma::mat with size P x (N + 1) (where P = 10).
							Filled with the weights connecting the last hidden layer to the output layer.
	diag_indices:			int vector of length D.
							Contains the indices n'_d for the nonzero diagonals of the hidden weight matrices.
	validation_batch_index:	int.
							Index of the validation batch. (Note that the first batch has the index 0.)
	epoch:					int.
							Index of the epoch. (Note that the first epoch has the index 0.)
	*/
	
	// print save diag indices to file
	string diag_file_name = "weights/diag_indices_val_batch_" + to_string(validation_batch_index + 1) + "_epoch_" + to_string(epoch + 1) + ".txt";
	ofstream diag_file;
	diag_file.open(diag_file_name);
	for (int n_prime : diag_indices){
		diag_file << n_prime << endl;
	}
	diag_file.close();
	
	// print input weights to file
	string input_file_name = "weights/input_weights_val_batch_" + to_string(validation_batch_index + 1) + "_epoch_" + to_string(epoch + 1) + ".txt";
	input_weights.save(input_file_name, csv_ascii);
	
	// print hidden weights to file
	string hidden_file_name = "weights/hidden_weights_val_batch_" + to_string(validation_batch_index + 1) + "_epoch_" + to_string(epoch + 1) + ".txt";
	hidden_weights.save(hidden_file_name, raw_ascii);
	
	// print output weights to file
	string output_file_name = "weights/output_weights_val_batch_" + to_string(validation_batch_index + 1) + "_epoch_" + to_string(epoch + 1) + ".txt";
	output_weights.save(output_file_name, csv_ascii);
}
