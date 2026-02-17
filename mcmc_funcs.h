#ifndef MCMC_FUNCS_H
#define MCMC_FUNCS_H

#include <gsl/gsl_rng.h>

// Computes the new positions of a metrolopis step
void metropolis_step(double* step_size, int N_wlkrs, int D, double* positions, double* new_positions);

// Accepts or declines the steps, updates the positions and saves them 
void accept_step(int N, int D, double* positions, double* new_positions, 
                 double* P, double* new_P, double* track, int* track_index);

// Generates a uniformly distributed random number in the range [min, max]
double uniform_range(double min, double max, gsl_rng *r);

#endif

