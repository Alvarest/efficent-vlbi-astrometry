# Efficent VLBI Astrometry
## Introduction
This repository contains optimized C code designed for performing high-precision astrometry on 
Very Long Base Interferometry (VLBI) data without a GPU. The implementation is tailored for 
accurate measurement of gravitational lens image positions, although it could be adapted to 
perform any other astrometry task.

The algorithm used is Markov-Chain Monte Carlo (MCMC) with the Metropolis method to compute steps. 
This approach is ideal for applications requiring statistical rigor and accurate results.

## Compilation and execution
### Prerequisites
To compile and run the code, ensure the following libraries and headers are installed in your 
system:
- GNU Scientific Library (GSL)
- OpenMP
You can install the required dependecies on **Ubuntu/Debian** with:
```
sudo apt install libgsl-dev libomp-dev
```
### Compilation
To compile the code, use the following gcc command on the root directory:
```
gcc -c mcmc_funcs.c -I. -lm -lgsl -lgslcblas -lgomp -fopenmp
gcc -c model_funcs.c -I. -lm -lgsl -lgslcblas -lgomp -fopenmp
```
### Execution
To link and compile your own code using this functions you should link the object files with 
your_code.c:
```
gcc -o your_code your_code.c mcmc_funcs.o model_funcs.o -lm -lgsl -lgslcblas -lgomp -fopenmp
```

