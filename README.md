# Usage of task and data parallelism for finding the lower boundary vectors in a stochastic-flow network

This repository contains the code and the inputs used in the paper 
https://www.sciencedirect.com/science/article/pii/S0951832023003319

To compile the code, use a recent version of `gcc` or `clang` and make sure you have OpenMP libraries installed.

```console
$ gcc -O3 d-mp.c -o d-mp -fopenmp
```

To execute, provide as input the network, d, the capacity to use for all the links (omit this parameter if you want to use the input file's capacities) and 0/1 to disable/enable vectorization.

For instance, the following command runs the algorithm on the network described by `23.csv` with d=3, cap=4 and vectorization on (1).

```console
$ ./d-mp benchs/23.csv 3 4 1
```
