// This is an example of how to use the set of functions to do precision astrometry off 
// radioastronomical VLBI data. This general structure can be modified to obtain any
// parameter of the observed objects with high presition and correct uncertainities.

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
#include "functions/mcmc_funcs.h"
#include "functions/model_funcs.h"

// Constants 
const double mas_to_radians = M_PI / 180.0 / 3600.0 / 1000.0;

const double RA_NE = mas_to_radians * 0.0;
const double DEC_NE = 0.0*mas_to_radians;
const double RA_SW = -641.95*mas_to_radians;  // milli-arcseconds
const double DEC_SW = -728.08*mas_to_radians; // milli-arcseconds
const double FRAT = 1.3;  // Amplitification ratio. 

const char trackfile_name[50] = "tracks/track_32w_1000s_0-4ss.dat";
const char datafile_name[50] = "data.dat";

// This functions must be defined by the user, as they depend on the previous knowledge
// we have of the source properties
void generate_init_positions(double* positions, int N, int D, const double* widths, const double* guess, gsl_rng *r);
int init_from_previous(double* positions, double* P, int N, int D, int N_steps, FILE* trackfile, 
                         const char* previous_trackfile_name);
void init_from_scratch(double *positions, double *P, const int N_steps, const int D, 
                       FILE *save, 
                       const double *widths, const double *guess, gsl_rng *r, 
                       const int N, const float *U, const float *V,     // length N
        			   const float *VISIB_R, const float *VISIB_I, // length N*F, row-major
            		   const float *WGT,                    // length N*F, row-major
            		   const double *LONG_OND,               // length F
    				   const int num_baselines, const int num_freqs, 
                       const double *norm_chi2, const double lim);

int main(){
    // Here we reserve memory and define basic data parameters to be used 
	int i;
	int j;

	double guess[5] = {RA_NE, DEC_NE, RA_SW, DEC_SW, FRAT};

	gsl_rng *r;

	FILE *datafile;
	FILE *save;

	int N_steps = 1000;
	int D = 5;
	int N = 32;
	double *positions = calloc(N * D, sizeof(double));
	double *new_positions = calloc(N * D, sizeof(double));
	double *P = calloc(N, sizeof(double));  
	double *new_P = calloc(N, sizeof(double));
	double step_size[5] = {
		0.4 * mas_to_radians,
		0.4 * mas_to_radians,
		0.4 * mas_to_radians,
		0.4 * mas_to_radians,
		0.1
	};

	int track_index = 0;
	int track_record = 80;
	int track_size = 100;
	double *track = calloc(track_size * N * D, sizeof(double));

	double width = 0.5 * mas_to_radians; 
	double widths[5] = {width, width, width, width, 0.5};
	double lim = 2 * mas_to_radians; 

	double *LONG_OND;
	float *U;
	float *V;
	float *VISIB_R;
	float *VISIB_I;
	float real, imag;
	float *WGT;
	int num_freqs;
	int num_baselines;
	double *p = calloc(5, sizeof(double));

	r = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(r, (unsigned long)time(NULL));

	// Open data file
	datafile = fopen(datafile_name, "rb");
	if (!datafile) {
		perror("Error while opening data.dat");
		return 1;
	}
	
	// Open save file
	save = fopen(trackfile_name, "wb");
	if (!save) {
		perror("Error while opening track.");
		return 1;
	}

	// Read the data
	printf("Reading data...");
    
	if (read_data(datafile, &LONG_OND, &U, &V, &VISIB_R, &VISIB_I, &WGT, &num_freqs, &num_baselines) != 0){
		fprintf(stderr, "Error leyendo data.dat\n");
		return 1;
	}
	fclose(datafile);
	
    // Computing a normalization constant
	double *norm_chi2 = calloc(num_baselines, sizeof(double));
	for (i = 0; i < num_baselines; i = i+1){
		double sum = 0.0;
		for (j = 0; j < num_freqs; j = j+1){
			sum = sum + *(VISIB_R + i*num_freqs + j) * *(VISIB_R + i*num_freqs + j) +
				*(VISIB_I + i*num_freqs + j) * *(VISIB_I + i*num_freqs + j);		
		}
		*(norm_chi2 + i) = sum;
	}
    
	printf(" data read correctly\n");

    init_from_scratch(positions, P, N_steps, D, save, widths, guess, r,
                       N, U, V, VISIB_R, VISIB_I, WGT, LONG_OND,
    				   num_baselines, num_freqs, norm_chi2, lim);

	// MCMC starts
	printf("\nStart.\n");
	double start = omp_get_wtime();
	double end;	

	i = 0;
	while (i < N_steps) {
		// New positions
		metropolis_step(step_size, N, D, positions, new_positions);

		// Probabilities of the new positions
		#pragma omp parallel for private(j)
		for (j = 0; j < N; j = j+1){
			*(new_P + j) = get_prob((new_positions + j*D), U, V, VISIB_R, VISIB_I, WGT, LONG_OND, 
						  num_baselines, num_freqs, norm_chi2, lim, guess);
		}

		// Accept and decline
		accept_step(N, D, positions, new_positions, P, new_P, track, &track_index);

		i = i+1;
		if (i % track_record == 0){
			fwrite(track, sizeof(double), track_record * N * D, save);	
			track_index = 0;
			end = omp_get_wtime();
			printf("Track saved. Iteration number %i. Execution time: %g.\n",
					i, (end - start));
		}
	}
	fwrite(track, sizeof(double), (i % track_record) * N * D, save);
	fwrite(P, sizeof(double), N, save);
	fclose(save);
	end = omp_get_wtime();
	printf("Execution time: %g seconds.\n", (end - start));

	free(positions);
	free(new_positions);
	free(P);
	free(new_P);
	free(track);
	free(norm_chi2);
	free(p);
	free(LONG_OND);
	free(U);
	free(V);
	free(VISIB_R);
	free(VISIB_I);
	free(WGT);
	return 0;
}


void generate_init_positions(double* positions, int N, int D, const double* widths, const double* guess, gsl_rng *r){
	int i; // Walkers
	int j; // Dimensions
	for (i = 0; i < N; i = i+1){
		for (j = 0; j < D; j = j+1){
			*(positions + D*i + j) = uniform_range(guess[j] - widths[j], guess[j] + widths[j], r);
		}
	}
}

int init_from_previous(double* positions, double* P, int N, int D, int N_steps, FILE* trackfile, 
                       const char* previous_trackfile_name){
    int step_ind;
    int dimension_ind;
    int walker_ind;
    
    int N_steps_prev;
    int D_prev;
    int N_prev;

    double *position_read = malloc(sizeof(double) * D);
    
    FILE *previous_trackfile = fopen(previous_trackfile_name, "rb");
	if (!previous_trackfile) {
		perror("Error while opening initialization track file.");
		return 1;
    }

    fread(&N_steps_prev, sizeof(int), 1, previous_trackfile);
    fread(&D_prev, sizeof(int), 1, previous_trackfile);
    fread(&N_prev, sizeof(int), 1, previous_trackfile);
    if (D_prev != D) {
        perror("Position dimensions doesn't coincide");
        return 1;
    } else if (N_prev != N) {
        perror("Number of walkers doesn't coincide");
        return 1;
    }
    N_steps_prev = N_steps_prev + N_steps;
    
    fwrite(&N_steps_prev, sizeof(int), 1, trackfile);
	fwrite(&D, sizeof(int), 1, trackfile);
	fwrite(&N, sizeof(int), 1, trackfile);

    // Copying previuos track
    for (step_ind = 0; step_ind < N_steps_prev-1; step_ind = step_ind + 1){
        for (walker_ind = 0; walker_ind < N; walker_ind = walker_ind + 1){
            fread(position_read, sizeof(double), D, previous_trackfile);
        	fwrite(position_read, sizeof(double), D, trackfile);
        }
    }

    // Copying last step to initialize the positions
    for (walker_ind = 0; walker_ind < N; walker_ind = walker_ind + 1){
        fread(position_read, sizeof(double), D, previous_trackfile);
        fwrite(position_read, sizeof(double), D, trackfile);
        for (dimension_ind = 0; dimension_ind < D; dimension_ind = dimension_ind + 1){
            positions[walker_ind * D + dimension_ind] = position_read[dimension_ind];
        }
    }

    // Reading last positions probabilities and saving them to a file
    fread(P, sizeof(double), N, previous_trackfile);
    fclose(previous_trackfile);  

    return 0;
}

void init_from_scratch(double *positions, double *P, const int N_steps, const int D,
                       FILE *save, 
                       const double *widths, const double *guess, gsl_rng *r, 
                       const int N, const float *U, const float *V,     // length N
        			   const float *VISIB_R, const float *VISIB_I, // length N*F, row-major
            		   const float *WGT,                    // length N*F, row-major
            		   const double *LONG_OND,               // length F
    				   const int num_baselines, const int num_freqs, 
                       const double *norm_chi2, const double lim){
    int i;
    
    generate_init_positions(positions, N, D, widths, guess, r);
    
	// First we save track shape
	fwrite(&N_steps, sizeof(int), 1, save);
	fwrite(&D, sizeof(int), 1, save);
	fwrite(&N, sizeof(int), 1, save);

	// Computing first positions probabilities
	#pragma omp parallel for private(i)
	for (i = 0; i < N; i = i+1){
		*(P+i) = get_prob((positions + i*D), U, V, VISIB_R, VISIB_I, WGT, LONG_OND, 
						  num_baselines, num_freqs, norm_chi2, lim, guess);
	}
}
