#ifndef F_H
#define F_H

#include <iostream>
#include <math.h>
#include <armadillo>




#ifndef USE_COS_SQUARED   //if not then use sin
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
#else  //else clause for cosine squared nonlinearity
static inline double f(double x){
    return std::pow(std::cos(x), 2);
}

static inline double f_prime(double x){
    return -2 * std::sin(x) * std::cos(x);  //maybe faster to rewrite in a different trigonometric form?
}

static inline arma::mat f_matrix(arma::mat &x){
    return arma::pow(arma::cos(x), 2);
}

static inline arma::mat f_prime_matrix(arma::mat &x){
    return -2 * arma::sin(x) % arma::cos(x);  //maybe faster to rewrite in a different trigonometric form?
}
#endif  //end of ifndef for switching the nonlinearity








// function g (input preprocessing function)
static inline double input_processing(double x){
    return 0.5 * M_PI * tanh(x);
}

static inline double input_processing_prime(double x){
    return 0.5 * M_PI / (cosh(x) * cosh(x));
}

#endif  
