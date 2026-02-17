#ifndef FUNCS_H
#define FUNCS_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <time.h>
#include "mcmc_funcs.h"
#include "model_funcs.h"

extern double ref_10std_max;
extern double ref_10std_min;

double fix_prob(double chi2); 

// Functions to print data 
void print_array(double* array, int length);
void float_print_array(float* array, int length);

// Functions to read observation data
void get_props(int int_size, int *dimension, int *tipo, int *dim, FILE *dataread);
int read_data(FILE *datafile, double **LONG_OND, float **U, float **V, float **VISIB_R, 
		float **VISIB_I, float **WGT, int *num_freqs, int *num_baselines);

// Functions to obtain the probability
static int cmp_double(const void *a, const void *b);
double compute_chisq_blocked_omp(
    double p[5],
    const float *U, const float *V,     // length N
    const float *VIS_RE, const float *VIS_IM, // length N*F, row-major
    const float *WGT,                    // length N*F, row-major
    const double *LONG_OND,               // length F
    const int N, const int F, const double *normalize
);
double prior(const double *p, const double lim, const double* guess);
double get_prob(double p[5],
				const float *U, const float *V,     // length N
				const float *VIS_RE, const float *VIS_IM, // length N*F, row-major
				const float *WGT,                    // length N*F, row-major
				const double *LONG_OND,               // length F
				const int N, const int F, const double *normalize,
				const double lim, const double* guess);

#endif

