"""
In this file we compare several different Markov Chain Monte Carlo algorithms on
an example where the target density has a volcano form. measure of comparison is
estimated effective sample size. The experiments are performed in several high
dimensional situations with the goal to see any tendencies of performace of the
algorithms when dimension increases.
"""

# %% imports
# standard library imports
import datetime
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
# set of the algorithms to test
algorithms = {"ESS" : mcmc.random_ESS,
              "RWM" : mcmc.random_RWM,
              "pCN" : mcmc.random_pCN}
# define the test function for autocorrelation function and effective sample size
@njit
def test_func(x):
    """Calculate log(1+|x|)."""
    return np.log(1 + np.linalg.norm(x))
# define Gaussian mixture with two peaks centered in
# m1 = (0, ..., 0) and m2 = (1, ..., 1)
# define standard deviations
sd1 = 3
sd2 = 2
# define weights. must sum to 1
weight1 = 0.5
weight2 = 1 - weight1

# define multivariate gaussian PDF
@njit
def multivariate_normal_pdf(x, mean, sd):
    d = len(x)
    norm_constant = sd**d
    # return -0.5 * np.linalg.norm(x - mean)**2 / sd**2 - d * np.log(sd)
    return np.exp(-0.5 * np.linalg.norm(x - mean)**2 / sd**2) / norm_constant

# define log of the PDF of the gaussian mixture
# @njit
# def ln_pdf(x):
#     d = len(x)
#     # define the means
#     m1 = np.zeros(d)
#     m1[0] = d
#     m2 = -m1 / 2
#     # m2 = -m1
#     # define two gaussians
#     g1 = multivariate_normal_pdf(x, m1, sd1)
#     g2 = multivariate_normal_pdf(x, m2, sd2)

#     return np.log(weight1 * g1 + weight2 * g2)

# define log of the volcano density function
@njit
def ln_pdf(x):
    """Calculate log value of the density at the point x."""
    norm_x = np.linalg.norm(x)
    return norm_x - 0.5 * norm_x**2

# define arguments for all dimensions and all algorithms
args = {}
for d in d_range:
    x0 = np.zeros(d)
    # x0 for guassian mixture
    x0[0] = d
    args[d] = {}
    for key in algorithms.keys():
        args[d][key] = {"ln_pdf"   : ln_pdf,
                        "size"     : N,
                        "burn_in"  : burn_in,
                        "x0"       : x0}
                        # "test_func": test_func}
        if key == "ESS" or key == "pCN":
            args[d][key]["sd"] = np.sqrt(2)
        if key == "RWM":
            args[d][key]["sd"] = 2.4 / np.sqrt(d)
        if key == "pCN":
            # angle_par for the volcano density
            # args[d][key]["angle_par"] = 1.5
            # angle_par for gaussian mixture with weights 0.5, 0.5
            # args[d][key]["angle_par"] = np.pi + 2.4 / d
            # angle_par for gaussian mixture with weights 0.8, 0.2
            # args[d][key]["angle_par"] = np.pi + 1.9 / d
            # angle_par for gaussian mixture with weights 0.5, 0.5 nonsymmetrical peaks
            # args[d][key]["angle_par"] = 2.4 / d
            # angle_par for close mixture
            # args[d][key]["angle_par"] = 2.4 / d
            # angle par for ICML review
            args[d][key]["angle_par"] = 4 / np.sqrt(d)

# %% calculating ACF for test_func
# set random seed
np.random.seed(1)
full_time = time.time()
acfs = {}
for d in d_range:
    print("Start computing for d =", d)
    # starting vector of dimension "d"
    x0 = np.zeros(d)
    # x0[0] = d
    acfs[d] = {}
    for key in algorithms.keys():        
        start_time = time.time()
        print(f"{key} started.")
        samples = algorithms[key](**args[d][key])
        print(f"{key} time: {datetime.timedelta(seconds=time.time() - start_time)}")
        acfs[d][key] = stattools.acf(samples, nlags=k_max, fft=True)[1:]

print(f"CPU time total = {datetime.timedelta(seconds=time.time() - full_time)}")

# %% calculating effective sample size
ess = {}
for alg in algorithms.keys():
    ess[alg] = [N / (1 + 2 * np.sum(acfs[d][alg])) for d in d_range]
# ess["ESS"] = np.array(ess["ESS"]) / (-1.5 + 3*np.log10(np.array(ess["ESS"])))
# %% save the kernel state
# import dill
# dill.dump_session("sss_vs_ess_kernel.db")

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

# %% simulating and drawing one path of the first coordinate of each algorithm
# d = 10

# print(f"start for d = {d}. x0 is the positive peak. {N} steps")
# mcmc.sample_and_draw_path(mcmc.random_ESS, "ESS", **args[d]["ESS"])
# # mcmc.sample_and_draw_path(mcmc.random_RWM, "RWM", **args[d]["RWM"])
# mcmc.sample_and_draw_path(mcmc.random_pCN, "pCN", **args[d]["pCN"])