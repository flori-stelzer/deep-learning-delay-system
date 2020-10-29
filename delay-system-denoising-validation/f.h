#ifndef F_H
#define F_H

#include <iostream>
#include <math.h>
#include <armadillo>

static inline double f(double x){
    return sin(x);
}

static inline double f_prime(double x){
    return cos(x);
}

static inline arma::mat f_matrix(arma::mat &x){
    return sin(x);
}

static inline arma::mat f_prime_matrix(arma::mat &x){
    return cos(x);
}

// function g (input preprocessing function)
static inline double input_processing(double x){
    return 0.5 * M_PI * tanh(x);
}

static inline double input_processing_prime(double x){
    return 0.5 * M_PI / (cosh(x) * cosh(x));
}

#endif  
