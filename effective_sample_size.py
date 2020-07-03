# %% imports
# standard library imports
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

# third party imports
from numba import njit
from statsmodels.tsa import stattools

# local application imports
import mcmc

# %% set initial parameters
# dimension
d_range = [10, 30, 100, 300, 1000]
# number of skipped iterations at the beginning
burn_in = 10**5
# number of iterations
N = 10**6
# the last k for calculating the autocorrelation function
k_max = 10**4

# define the density function
@njit
def density_func(x):
    """Calculate value of the density at the point x."""
    norm_x = np.linalg.norm(x)
    return np.exp(norm_x - 0.5 * norm_x**2)

# define inverse of the denisty_func. Needed for e.g. simple slice sampler
@njit
def density_inv(t):
    """Calculate the inverse of the density_func."""
    return 1 + np.sqrt(1 - 2*np.log(t))

# define the test function for autocorrelation function and effective sample size
@njit
def test_func(x):
    """Calculate log(1+|x|)."""
    return np.log(1 + np.linalg.norm(x))

# %% set of the algorithms
algorithms = {"ESS" : mcmc.random_ESS,
              "RWM" : mcmc.random_RWM,
              "pCN" : mcmc.random_pCN}

# %% calculating ACF for test_func
# set random seed
np.random.seed(1)
full_time = time.time()
acfs = {}
for d in d_range:
    print("Start computing for d =", d)
    # starting vector of dimension "d"
    x0 = np.zeros((d,))
    acfs[d] = {}
    for key in algorithms.keys():
        start_time = time.time()
        print(f"{key} started.")
        samples = algorithms[key](density_func, N, burn_in, x0, test_func=test_func)
        print(f"{key} time: {time.time() - start_time}")
        acfs[d][key] = stattools.acf(samples, nlags=k_max, fft=True)[1:]

print(f"CPU time total = {time.time() - full_time}")

# %% calculating effective sample size
ess = {}
for alg in algorithms.keys():
    ess[alg] = [N / (1 + 2 * np.sum(acfs[d][alg])) for d in d_range]

# %% save the kernel state
import dill
dill.dump_session("sss_vs_ess_kernel.db")

# %% load the kernel state if needed
# import dill
# dill.load_session("sss_vs_ess_kernel.db")

# %% plot ACF
for d in d_range:
    for alg in algorithms.keys():
        plt.plot(range(1, k_max + 1), acfs[d][alg])
    plt.title(f"{d} dimensional ACF")
    plt.xscale("log")
    plt.xlabel("Lag")
    plt.legend(algorithms.keys())
    plt.savefig(f"pics/ACF_d{d}.pdf")
    plt.show()

# %% plot effective sample size
for alg in algorithms.keys():
# for alg in ["SSS", "iESS"]:
    plt.plot(d_range, ess[alg], '-o')
plt.title("Effective sample size")
plt.xlabel("Dimension")
plt.xscale("log")
plt.yscale("log")
plt.legend(algorithms.keys())
plt.savefig("pics/ESS.pdf")
plt.show()

# %% plot corrected effective sample size
# for alg in algorithms.keys():
#     if alg == "ESS corrected": alg = "ESS"
#     plt.plot(d_range, ess[alg], '-o')
# plt.title("Effective sample size corrected")
# plt.xlabel("Dimension")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend(algorithms.keys())
# plt.savefig("pics/ESS_corrected.pdf")
# plt.show()

# %% test algorithms
# x0 = np.zeros((2,))
# test_SSS  = random_SSS (N, burn_in, x0)
# test_iESS = random_iESS(N, burn_in, x0)
# test_ESS  = random_ESS (N, burn_in, x0)
# test_RWM  = random_RWM (N, burn_in, x0)
# test_pCN  = random_pCN (N, burn_in, x0)


# %% drawing one histogram of the first coordinates with the target density
# values = [test_SSS, test_iESS, test_ESS, test_RWM, test_pCN]
# count, bins, ignored = plt.hist(values, bins=20, density=True)
# plt.legend(algorithms.keys())
# plt.show()

# %% drawing separate histograms of the first coordinates with the target density
# draw_histogram_check(test_SSS,  "SSS")
# draw_histogram_check(test_iESS, "iESS")
# draw_histogram_check(test_ESS,  "ESS")
# draw_histogram_check(test_RWM,  "RWM")
# draw_histogram_check(test_pCN,  "pCN")
# h1 = plt.hist2d(test_SSS [:, 0], test_SSS [:, 1], bins=50)
# h2 = plt.hist2d(test_iESS[:, 0], test_iESS[:, 1], bins=50)
# h3 = plt.hist2d(test_RWM [:, 0], test_RWM [:, 1], bins=50)
# h4 = plt.hist2d(test_pCN [:, 0], test_pCN [:, 1], bins=50)


# %% simulating and drawing one path of the first coordinate of each algorithm
# sample_and_draw_path(random_SSS,  labels["SSS"],  x0, 1000)
# sample_and_draw_path(random_iESS, labels["iESS"], x0, 1000)
# sample_and_draw_path(random_ESS, "ESS", np.zeros((10)), 1000)
# sample_and_draw_path(random_RWM2,  "RWM",  np.zeros(10), 1000)
# sample_and_draw_path(random_pCN,  labels["pCN"],  x0, 1000)

# %% tune acceptance probability of RWM
# burn_in = 10**4
# N = 10**5
# d = 10
# x0 = np.zeros(d)
# for par in np.arange(0.79, 0.91, 0.05):
#     print(f"par = {par}")
#     random_RWM(par, N, burn_in, x0, test_func, print_time=False)
