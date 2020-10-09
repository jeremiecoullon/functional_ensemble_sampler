#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void euler_Langevin(const double *ww, double *xx, double *pp, const double alpha, const double sigma, const size_t nn, const double dt){
  // Euler solver for Langevin dynamics in a harmonic potential
  for (size_t i=1; i<nn; i++){
    xx[i] = xx[i-1] + pp[i-1]*dt;
    pp[i] = pp[i-1] - alpha*xx[i]*dt + sigma*(ww[i] - ww[i-1]);
  }
}
