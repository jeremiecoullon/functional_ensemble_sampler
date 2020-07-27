#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void euler_solver_fun(const double *ww, double *pp, const double theta, const double sigma, const size_t nn, const double dt){
  // solver for OU problem
  for (size_t i=1; i<nn; i++){
    pp[i] = pp[i-1] - theta*pp[i-1]*dt + sigma*(ww[i] - ww[i-1]);
  }
}

void euler_double_well(const double *ww, double *pp, const size_t nn, const double dt){
  // solver for BM in a double well potential
  for (size_t i=1; i<nn; i++){
    pp[i] = pp[i-1] + 10*pp[i-1]*(1-pow(pp[i-1], 2))/(1+pow(pp[i-1], 2))*dt + ww[i] - ww[i-1];
  }
}

void euler_Langevin(const double *ww, double *xx, double *pp, const double alpha, const double sigma, const size_t nn, const double dt){
  // Euler solver for Langevin dynamics in a harmonic potential
  for (size_t i=1; i<nn; i++){
    xx[i] = xx[i-1] + pp[i-1];
    pp[i] = pp[i-1] - alpha*xx[i]*dt + sigma*(ww[i] - ww[i-1]);
  }
}
