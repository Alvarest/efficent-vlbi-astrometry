#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <omp.h>
#include "mcmc_funcs.h"

// This is a set of functions for running MCMC with the Montecarlo algorithm
// To do so, you must follow this steps:
// 	1. Inicialize walkers values
// 	2. Compute the probabilities
// 	3. Step forward
// 	4. Compute new probabilities
// 	5. Accept or decline the step for each walker depending on the probabilities
// 	6. Repeat from 3 until you reach the desired number of iterations

void metropolis_step(double* step_size, int N_wlkrs, int D, double* positions, double* new_positions){
	int wlkr_id; 
	int dim; 
	gsl_rng **r_array = calloc(omp_get_max_threads(), sizeof(gsl_rng*)); 
	
	#pragma omp parallel private(wlkr_id,dim)
	{
		int tid = omp_get_thread_num();
		r_array[tid] = gsl_rng_alloc(gsl_rng_mt19937); 
		gsl_rng_set(r_array[tid], (unsigned long)(time(NULL) + tid)); 
		for (wlkr_id = 0; wlkr_id < N_wlkrs; wlkr_id = wlkr_id+1){
			for (dim = 0; dim < D; dim = dim+1){
				*(new_positions + dim + D*wlrk_id) = *(positions + dim + D*wlrk_id) + 
					(gsl_rng_uniform(r_array[tid]) - 0.5) * *(step_size+dim);
			}
		}
	}
}

void accept_step(int N_wlkrs, int D, double* positions, double* new_positions, 
				 double* P, double* new_P, double* track, int* track_index){
	int wlkr_id; // Walkers
	double step_prob;
	double rnd_gen;

	int base = *track_index;
	*track_index = *track_index + N_wlkrs*D;

	gsl_rng **r_array = calloc(omp_get_max_threads(), sizeof(gsl_rng*)); 

	#pragma omp parallel private(wlkr_id, step_prob, rnd_gen)
	{
		int tid = omp_get_thread_num();
		r_array[tid] = gsl_rng_alloc(gsl_rng_mt19937); 
		gsl_rng_set(r_array[tid], (unsigned long)(time(NULL) + tid)); 

		for (wlkr_id = 0; wlkr_id < N_wlkrs; wlkr_id++) {
			if (*(P+wlkr_id) == 0) {
				step_prob = 1;
			} else {
				step_prob = *(new_P+wlkr_id) / *(P+wlkr_id);
			}
			rnd_gen = gsl_rng_uniform(r_array[tid]);
			if (rnd_gen < step_prob) {
				for (int dim = 0; dim < D; dim = dim+1) {
					*(positions + dim + D*wlkr_id) = *(new_positions + dim + D*wlkr_id);
				}
				*(P+wlkr_id) = *(new_P + wlkr_id);
			}
			for (int dim = 0; dim < D; dim = dim+1) {
				*(track + base + wlkr_id*D + dim) = *(positions + wlkr_id*D + dim);
			}
		}
		gsl_rng_free(r_array[tid]);
	}
	
	free(r_array);
}

double uniform_range(double min, double max, gsl_rng *r){
	return (gsl_rng_uniform(r) * (max - min)) + min;
}
