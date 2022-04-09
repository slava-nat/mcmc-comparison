# Description
In this project we perform numerical experiments with Python for the following Markov chain Monte Carlo algorithms:

* Random Walk Metropolis,
* Simple Slice Sampling,
* Elliptical Slice Sampling,
* Preconditioned Crank-Nicolson

as it was done in the following paper:

> Natarovskii, V., Rudolf, D., and Sprungk, B. (2021).
Geometric convergence of elliptical slice sampling.
In *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 7969–7978.
> [https://proceedings.mlr.press/v139/natarovskii21a.html](https://proceedings.mlr.press/v139/natarovskii21a.html)

# File structure

* `mcmc.py` - In this module you can find functions for generating samples by using different Markov chain Monte Carlo methods (MCMC), such as:
    * Random Walk Metropolis,
    * Simple Slice Sampler,
    * Elliptical Slice Sampler,
    * Preconditioned Crank-Nicolson.

* `main_algorithms_comparison.py` - In this file we compare several different Markov Chain Monte Carlo algorithms on an example where the target density has a volcano form.
Measure of comparison is the estimated effective sample size.
The experiments are performed in several high dimensional situations with the goal to see any tendencies of performance of the algorithms when dimension increases.

* `main_double_banana.py` - In this file we check how Elliptical Slice Sampler performs on an example where the target density is of "double banana" form.

* `main_not_lower_semicont.py` - In this fie we investigate Elliptical Slice Sampler on an example where the target density is an indicator function of a d-dimensional cube [0,1]^d.
This kind of density is not lower semicontinous. Our hypothesis is that lower semicontinuity is the minimum requirement for Elliptical Slice Sampler to be reversible.
The challenging starting point in this example is x0=0, because then with positive probability one stays at this point and cannot escape it.

# Dependencies

An open source library [Numba](https://numba.pydata.org/) is used for achieving better performance.
You can install Numba using [this guide](https://numba.readthedocs.io/en/stable/user/installing.html).