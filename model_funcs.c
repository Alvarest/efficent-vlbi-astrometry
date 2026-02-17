// This file contains a set of functions to perform highly efficent astrometry on 
// radioastronomical VLBI data using the MCMC method with the Montecarlo algorithm.

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

double ref_10std_max = 151638.34;
double ref_10std_min = 108557.88;

double fix_prob(double chi2){
	return (chi2 - ref_10std_max) / (ref_10std_min - ref_10std_max);
}

// Functions to print data 
void print_array(double* array, int length){
	int i;
	printf("[");
	for (i = 0; i < length - 1; i = i+1){
		printf("%g ", *(array + i));
	}
	printf("%g]\n", *(array + length - 1));
}
void float_print_array(float* array, int length){
	int i;
	printf("[");
	for (i = 0; i < length - 1; i = i+1){
		printf("%g ", *(array + i));
	}
	printf("%g]\n", *(array + length - 1));
}

// Functions to read observation data
void get_props(int int_size, int *dimension, int *tipo, int *dim, FILE *dataread){
	int i;
	int mock;

	fread(dimension, int_size, 1, dataread);
	fread(tipo, int_size, 1, dataread);
	*dim = 1;
	for (i = 0; i < *dimension; i = i+1){
		fread(&mock, int_size, 1, dataread);
		*dim = *dim * mock;
	}
}

int read_data(FILE *datafile, double **LONG_OND, float **U, float **V, float **VISIB_R, 
		float **VISIB_I, float **WGT, int *num_freqs, int *num_baselines){
	int float_size = sizeof(float);
	int double_size = sizeof(double);
	int int_size = sizeof(int);
	int *dimension = calloc(1, int_size);
	int *tipo = calloc(1, int_size);
	int *dim = calloc(1, int_size);
	float real;
	float imag;

	int i;

	if (datafile == NULL) {
		printf("Error abriendo el archivo.\n");
		return 1;

	} else {
		// LONG_OND
		get_props(int_size, dimension, tipo, dim, datafile);	
		*LONG_OND = calloc(*dim, double_size);
		for (i = 0; i < *dim; i = i+1){
			fread((*LONG_OND + i), double_size, 1, datafile); 
		}
		*num_freqs = *dim;

		for (i = 0; i < *dim; i = i+1){
			*(*LONG_OND + i) = 2 * M_PI / *(*LONG_OND + i);
		}

		// U
		get_props(int_size, dimension, tipo, dim, datafile);	
		*U = calloc(*dim, sizeof(float));
		fread(*U, float_size, *dim, datafile);
		*num_baselines = *dim;
		
		// V
		get_props(int_size, dimension, tipo, dim, datafile);	
		*V = calloc(*dim, sizeof(float));
		fread(*V, float_size, *dim, datafile);

		// VISIB
		get_props(int_size, dimension, tipo, dim, datafile);	
		*VISIB_R = calloc(*dim, sizeof(float));
		*VISIB_I = calloc(*dim, sizeof(float));
		for (i = 0; i < *dim; i = i+1){
			fread(*VISIB_R + i, float_size, 1, datafile);
			fread(*VISIB_I + i, float_size, 1, datafile);
		}

		// WGT
		get_props(int_size, dimension, tipo, dim, datafile);
		*WGT = calloc(*dim, float_size);
		for (i = 0; i < *dim; i = i + 1){
			fread((*WGT + i), float_size, 1, datafile); 
		}
		return 0;
	}
}


// Functions to obtain the probability
static int cmp_double(const void *a, const void *b){
	double va = *(const double*)a;
	double vb = *(const double*)b;
	return (va > vb) - (va < vb);
}

static void normalize_abs(double complex *model, const int i, const int F){
	double abs_mean;
	abs_mean = 0;
	for (int j = 0; j < F; j = j+1){
		abs_mean = abs_mean + cabs(*(model + j));
	}
	abs_mean = abs_mean / F;

	for (int j = 0; j < F; j = j+1){
		*(model + j) = *(model + j) / abs_mean;
	}
}

static double calc_arg_mean(double complex *model, const int i, const int F){
	double arg_mean = 0;
	for (int j = 0; j < F; j = j+1){
		arg_mean = arg_mean + carg(*(model + j));
	}
	arg_mean = arg_mean / F;

	return arg_mean;
}

double compute_chisq_blocked_omp(
		double p[5],
		const float *U, const float *V,
		const float *VIS_RE, const float *VIS_IM,
		const float *WGT,
		const double *LONG_OND,
		const int N, const int F, const double *normalize
){
	const double lr1 = p[0], mr1 = p[1], lr2 = p[2], mr2 = p[3], fr = p[4];
	double chi2_total = 0.0;

	int l;
	printf("[");
	for (l = 0; l < F-1; l = l+1){
		printf("%g,", VIS_RE[l]);
	}
	printf("%g]\n", VIS_RE[F]);
	
	#pragma omp parallel
	{
		double *tmp = (double*) malloc(sizeof(double) * F);
		double complex *model = (double complex*) malloc(sizeof(double complex) * F);
		if (!tmp || !model) {
			fprintf(stderr, "Fallo en la reserva de memoria.\n");
			#pragma omp critical
			{ exit(1); }
		}

		#pragma omp for schedule(static) reduction(+:chi2_total)
		for (int j = 0; j < N; j = j+1){
			double fuente1 = U[j] * lr1 + V[j] * mr1;
			double fuente2 = U[j] * lr2 + V[j] * mr2;

			for (int i = 0; i < F; i = i+1) {
				double ang1 = LONG_OND[i] * fuente1;
				double ang2 = LONG_OND[i] * fuente2;
				double r1 = fr * cos(ang1);
				double im1 = fr * sin(ang1);
				double r2 = cos(ang2);
				double im2 = sin(ang2);
				model[i] = r1 + r2 + I * (im1 + im2);
			}

			normalize_abs(model, j, F);

			double arg_mean;
			arg_mean = calc_arg_mean(model, j, F);
			double cosph = cos(-arg_mean), sinph = sin(-arg_mean);

			double chi2_row = 0.0;
			int base_idx = j * F;
			for (int i = 0; i < F; i = i+1) {
				double real_m = creal(model[i]) * cosph - cimag(model[i]) * sinph;
				double imag_m = creal(model[i]) * sinph + cimag(model[i]) * cosph;

				double real_o = (double)VIS_RE[base_idx + i];
				double imag_o = (double)VIS_IM[base_idx + i];

				double diff_re = real_o - real_m;
				double diff_im = imag_o - imag_m;
				double d2 = diff_re * diff_re + diff_im * diff_im;
				chi2_row += d2 * (double)WGT[base_idx + i];
			}

			chi2_total += chi2_row / *(normalize+j);
		}

		free(tmp);
		free(model);
	}

	printf("\n");
	return chi2_total;
}

double prior(const double *p, const double lim, const double* guess){
	double RA_NE = guess[0];
	double DEC_NE = guess[1];
	double RA_SW = guess[2];
	double DEC_SW = guess[3];
	double FRAT = guess[4];
	double rlr1 = p[0] - RA_NE;
	double rlm1 = p[1] - DEC_NE;
	double rlr2 = p[2] - RA_SW;
	double rlm2 = p[3] - DEC_SW;
	double rfr = p[4] - FRAT;

	int i;
	for (i = 0; i < 5; i = i + 1) {
		if (fabs(p[i] - guess[i]) > lim) {
			return 0.0;
		}
	}

	return 1.0;
}

double get_prob(double p[5],
				const float *U, const float *V,     // length N
				const float *VIS_RE, const float *VIS_IM, // length N*F, row-major
				const float *WGT,                    // length N*F, row-major
				const double *LONG_OND,               // length F
				const int N, const int F, const double *normalize,
				const double lim, const double* guess){

	double chi2 = compute_chisq_blocked_omp(p, U, V, VIS_RE, VIS_IM, WGT, LONG_OND, N, F, normalize);
	printf("chi2 = %g\n", chi2);
	printf("fix_prob = %g\n", fix_prob(chi2));
	return fix_prob(chi2) * prior(p, lim, guess);

}

