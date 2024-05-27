# Nesting Importance Sampling

The supplementary material for the ICML'24 paper "Estimating the Permanent by Nesting Importance Sampling". The repository contains the source files required for compiling the estimators.

## Compiling

Compilation requires C++ library Boost and the file `Breal.hpp`, which is included. You can compile, for example, the DeepNIS scheme with

```
g++ deepnis.cpp -o deepnis
```

## Input format

The input is given in the following format:

```
n time_limit
A_11  ..  ..  A_1n
 ..   ..       ..
 ..       ..   ..
A_n1  ..  ..  A_nn
```

Additionally, you can give the desired accuracy as command line arguments for the program, that is, `./deepnis [epsilon] [delta]`. The default arguments are `epsilon = 0.01` and `delta = 0.05`.

## Output format

Both estimators output debugging information to standard error stream and the actual output into the standard output stream in the following format:

```
log_estimate time_used
```

## deepnis.cpp

Our nesting importance sampling scheme using the depth-16 variant of the extended Huber bound.

## deepar.cpp

The rejection sampling scheme of Harviainen and Koivisto (2023) with depth-16 variant of the extended Huber bound.

## block_exact.cpp

Exact computation of the permanents of matrices in the Block Diagonal class.

## bernoulli40.in

A sample matrix of size 40×40 whose entries are Bernoulli-distributed with parameter `0.1`.
