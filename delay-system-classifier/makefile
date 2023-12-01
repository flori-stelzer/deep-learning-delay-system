CXX=g++
CXXFLAGS=-Ofast -std=c++14 -Wall

SOURCEFILES=main.cpp helper_functions.cpp solve_network.cpp solve_dde.cpp backprop.cpp numerical_gradient.cpp
INCLUDES_AND_LINKING=-I /home/y60000/armadillo-12.4.1/include -DARMA_DONT_USE_WRAPPER -lopenblas -lhdf5

#needs modules: aquarius, hdf5
all_aquarius: fit_dnn_sin_val fit_dnn_sin_test fit_dnn_cos_squared_val fit_dnn_cos_squared_test

fit_dnn_sin_val: main.cpp $(SOURCEFILES)
	$(CXX) $(CXXFLAGS) -o fit_dnn_sin_val.x  $(SOURCEFILES) $(INCLUDES_AND_LINKING)

fit_dnn_sin_test: main.cpp $(SOURCEFILES)
	$(CXX) $(CXXFLAGS) -DTESTING -o fit_dnn_sin_test.x  $(SOURCEFILES) $(INCLUDES_AND_LINKING)

fit_dnn_cos_squared_val: main.cpp $(SOURCEFILES)
	$(CXX) $(CXXFLAGS) -DUSE_COS_SQUARED -o fit_dnn_cos_squared_val.x $(SOURCEFILES) $(INCLUDES_AND_LINKING)

fit_dnn_cos_squared_test: main.cpp $(SOURCEFILES)
	$(CXX) $(CXXFLAGS) -DUSE_COS_SQUARED -DTESTING -o fit_dnn_cos_squared_test.x $(SOURCEFILES) $(INCLUDES_AND_LINKING)

#needs modules: odyssey, fj, hdf5
fit_dnn_cos_squared_val_ody_fcc: main.cpp $(SOURCEFILES)
	FCCpx -DUSE_COS_SQUARED -Kfast -SSL2 -std=c++14 -o fit_dnn_cos_squared_val_fcc.x $(SOURCEFILES) $(INCLUDES_AND_LINKING)

fit_dnn_cos_squared_test_ody_fcc: main.cpp $(SOURCEFILES)
	FCCpx -DUSE_COS_SQUARED -DTESTING -Kfast -SSL2 -std=c++14 -o fit_dnn_cos_squared_test_fcc.x $(SOURCEFILES) $(INCLUDES_AND_LINKING)

fit_dnn_cos_squared_val_ody_gcc: main.cpp $(SOURCEFILES)
	g++ -Ofast -std=c++14 -o fit_dnn_cos_squared_val_gcc.x ${SOURCEFILES} -I /home/y60000/armadillo-12.4.1/include -DARMA_DONT_USE_WRAPPER -L/work/opt/local/x86_64/cores/intel/2023.0.0/compiler/latest/linux/lib/intel64/ -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lhdf5

clean:
	rm -f fit_dnn_sin_val.x fit_dnn_sin_test.x fit_dnn_cos_squared_val.x fit_dnn_cos_squared_test.x fit_dnn_cos_squared_test_fcc.x fit_dnn_cos_squared_val_fcc.x