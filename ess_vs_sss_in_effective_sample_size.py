# %% imports
import dill
import functools as ft
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

from numba import jit, njit, prange
from statsmodels.tsa import stattools

# %% functions
@njit
def pi(x):
    """Calculate value of the density at the point x."""
    norm_x = np.linalg.norm(x)
    return np.exp(norm_x - 0.5 * norm_x**2)

@njit
def pi2(x, y):
    """Basically pi(x) / pi(y)."""
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return np.exp(norm_x - norm_y + 0.5*(-norm_x**2  + norm_y**2))

@njit
def first_coordinate(x):
    return x[0]

@njit
def random_RWM(size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate, print_avg_acceptance_rate=True):
    """
    Perform the Maetropolis-Hastings Random Walk algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in. If "test_func" is given then return test_func(samples).
    """
    d = len(x0)
    sampled_values = np.zeros((size,))
    sum_acceptance_rates = 0
    for i in range(-burn_in, size):
        x1 = x0 + np.random.normal(0, scale=2.5/np.sqrt(d), size=d)
        acceptance_rate = min(1, pi2(x1, x0))
        x0 = x1 if np.random.uniform(0, 1) < acceptance_rate else x0
        # save values after burn_in
        if i >= 0:
            sum_acceptance_rates += acceptance_rate
            sampled_values[i] = test_func(x0)
    if print_avg_acceptance_rate:
        print("RWM: average acceptance rate = ", sum_acceptance_rates / size)
    return sampled_values

@njit
def runiform_disc(d, R=1, r=0):
    """
    Sample efficiently from a uniform distribution on a d-dimensional disc
    centered in zero D(R, r) = {x : r < |x| < R}.
    """
    x = np.random.normal(0, 1, size=d)
    if r == 0:
        u = np.random.uniform(0, 1)
        return R * u**(1/d) * x / np.linalg.norm(x)
    u = np.random.uniform(r**d, R**d)
    return u**(1/d) * x / np.linalg.norm(x)
    # u = np.random.uniform()
    # return (r**d + (R**d - r**d) * u)**(1/d) * x / np.linalg.norm(x)
    # u = np.random.uniform()
    # return (r + (R - r) * u**(1/d)) * x / np.linalg.norm(x)

# x = np.array([runiform_disc(2, 4, 0) for i in range(10000)])
# h = plt.hist2d(x[:, 0], x[:, 1], bins=50)

@njit
def random_SSS(size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate):
    """
    Perform the Simple Slice Sampler algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in. If "test_func" is given then return test_func(samples).
    """
    d = len(x0)
    sampled_values = np.zeros((size,))
    for i in range(-burn_in, size):
        t = np.random.uniform(0, pi(x0))
        R = 1 + np.sqrt(1 - 2*np.log(t))

        x0 = runiform_disc(d, R, max(0, 2 - R))
        # save values after burn_in
        if i >= 0: sampled_values[i] = test_func(x0)
    return sampled_values

@njit
def random_two_segments(left_border, right_border, shift=np.pi):
    """
    Sample from a uniform distribution on a union of two swgments:
    [left_border, right_border] and [left_border + shift, right_border + shift]
    """
    x = np.random.uniform(left_border, right_border)
    return x + shift if np.random.binomial(1, 0.5) == 1 else x

@njit
def ellipse_point(x1, x2, angle):
    """Return a point on the ellipse generated by x1 and x2 with the angle a."""
    # if not len(x1) == len(x2):
    #     sys.exit("ERROR in ellipse_point: lengths of x1 and x2 must be equal")
    return(x1 * np.cos(angle) + x2 * np.sin(angle))

@njit
def random_ESS(size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate):
    """
    Perform the Elliptical Slice Sampler algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in. If "test_func" is given then return test_func(samples).
    """
    d = len(x0)
    sampled_values = np.zeros((size,))
    for i in range(-burn_in, size):
        norm_x0 = np.linalg.norm(x0)
        t = np.random.uniform(0, np.exp(norm_x0))
        w = np.random.normal(0, 1, size=d)
        norm_w = np.linalg.norm(w)

        R = max(0, np.log(t))

        Ax = norm_x0**2 - norm_w**2
        Bx = 2 * np.sum(x0 * w)
        Cx = 2 * R**2 - norm_x0**2 - norm_w**2

        phi = np.sign(Bx) * np.arccos(Ax / np.sqrt(Ax**2 + Bx**2))
        psi = np.arccos(max(-1, Cx / np.sqrt(Ax**2 + Bx**2)))

        theta = random_two_segments((phi - psi) / 2, (phi + psi) / 2)

        x0 = ellipse_point(x0, w, theta)
        # save values after burn_in
        if i >= 0: sampled_values[i] = test_func(x0)
    return sampled_values

@njit
def random_pCN(size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate, print_avg_acceptance_rate=True):
    """
    Perform the Preconditioned Crank-Nicolson algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in.
    """
    d = len(x0)
    sampled_values = np.zeros((size,))
    sum_acceptance_rates = 0
    for i in prange(-burn_in, size):
        norm_x0 = np.linalg.norm(x0)
        w = np.random.normal(0, 1, size=d)

        x1 = ellipse_point(x0, w, 1.5)
        acceptance_rate = min(1, np.exp(-norm_x0 + np.linalg.norm(x1)))
        x0 = x1 if np.random.uniform(0, 1) < acceptance_rate else x0

        # save values after burn_in
        if i >= 0:
            sum_acceptance_rates += acceptance_rate
            sampled_values[i] = test_func(x0)
    if print_avg_acceptance_rate:
        print("pCN: average acceptance rate =", (sum_acceptance_rates / size))
    return sampled_values

def rho(x):
    """Basically normalized pi for d = 1."""
    return (np.exp(-abs(x) - 0.5 * x**2)) / 1.31136

def draw_histogram_check(samples, title, bins=50, range=[-3, 3]):
    """Draw histogramm with rho over it."""
    count, bins, ignored = plt.hist(samples,
                                    bins=bins, density=True, range=range)
    plt.title(title)
    # plt.plot(bins, rho(bins), color='r')
    # plt.plot(bins, np.exp(-abs(bins)) / 2, color='b')
    plt.show()

def sample_and_draw_path(algorithm, title, x0, steps):
    """
    Simulate the algorithm given number of steps starting from x0 and
    draw the path of the first coordinate.
    """
    plt.title(title)
    plt.plot(range(steps), algorithm(steps, 0, x0)[:, 0])
    plt.show()

@njit
def get_correlations(samples, k_range=range(1, 11)):
    """
    Calculate correlations of "samples" for all k in "k_range".
    Returns a vector of length len("k_range").
    """
    return [np.corrcoef(samples[:-k], samples[k:])[0, 1] for k in k_range]

def sample_and_get_acf(algorithms, size, burn_in, x0, test_func, k_max, print_time=True):
    """
    Sample w.r.t. all algorithms from the dictionary "algorithms" "N" times
    after "burn_in" and calculate ACF of "test_func" for the first "k_max" lags.
    Return the dictionary of ACFs with the keys from "algorithms".
    """
    acfs = dict()
    for key in algorithms.keys():
        start_time = time.time()
        if print_time: print(f"{key} started.")
        samples = algorithms[key](size, burn_in, x0, test_func=test_func)
        if print_time: print(f"{key} time: {time.time() - start_time}")
        acfs[key] = stattools.acf(samples, nlags=k_max, fft=True)[1:]
    return acfs

def effective_sample_size(acf, N):
    """Calculate effective sample size using "acf" of size "N"."""
    return N / (1 + 2 * np.sum(acf))

# %% set initial parameters
# dimension
d_range = [10, 30, 100, 300, 1000]
# d_range = [10]
# number of skipped iterations at the beginning
burn_in = 10**5
# number of iterations
N = 10**6
# the last k for calculating the autocorrelation function
k_max = 10**4
# define the test function for autocorrelation function and effective sample size
@njit
def test_func(x):
    """Calculate log(1+|x|)."""
    return np.log(1 + np.linalg.norm(x))

# %% set full names of the algorithms
algorithms = {"SSS" : random_SSS,
              "iESS": random_ESS,
              "RWM" : random_RWM,
              "pCN" : random_pCN}

# %% calculating ACF for test_func
# set random seed
np.random.seed(1)
acfs = {}
for d in d_range:
    print("Start computing for d =", d)
    # starting vector of dimension "d"
    x0 = np.zeros((d,))
    acfs[d] = {}
    for key in algorithms.keys():
        start_time = time.time()
        print(f"{key} started.")
        samples = algorithms[key](N, burn_in, x0, test_func=test_func)
        print(f"{key} time: {time.time() - start_time}")
        acfs[d][key] = stattools.acf(samples, nlags=k_max, fft=True)[1:]

    # acfs[d] = sample_and_get_acf(algorithms, N, burn_in, x0, test_func, k_max)

# %% plot ACF
for d in d_range:
    for alg in algorithms.keys():
        plt.plot(range(1, k_max + 1), acfs[d][alg])
    plt.title(f"{d} dimensional ACF for log(1+|x|)")
    plt.xscale("log")
    plt.xlabel("Dimension")
    plt.legend(algorithms.keys())
    plt.savefig(f"ACF_d{d}.pdf")
    plt.show()

# %% calculating effective sample size
ess = {}
for alg in algorithms.keys():
    ess[alg] = [effective_sample_size(acfs[d][alg], N) for d in d_range]

# %% plot effective sample size
for alg in algorithms.keys():
    plt.plot(d_range, ess[alg], '-o')
plt.title("Effective sample size")
plt.xlabel("Dimension")
plt.xscale("log")
plt.yscale("log")
plt.legend(algorithms.keys())
plt.savefig("ESS.pdf")
plt.show()

# %% start parallelizing
# print("parallelizing...")
# p_SSS = random_SSS.remote(N, burn_in, x0)
# p_ESS = random_ESS.remote(N, burn_in, x0)
# p_RWM = random_RWM.remote(N, burn_in, x0)
# p_pCN = random_pCN.remote(N, burn_in, x0)
#
# test_SSS, test_ESS, test_RWM, test_pCN = ray.get([p_SSS, p_ESS, p_RWM, p_pCN])
# print("Done.")
# ray.shutdown()

# %% test algorithms
# x0 = np.zeros((2,))
# test_SSS = random_SSS(N, burn_in, x0)
# test_ESS = random_ESS(N, burn_in, x0)
# test_RWM = random_RWM(N, burn_in, x0)
# test_pCN = random_pCN(N, burn_in, x0)

# %% save the kernel state
# dill.dump_session("sss_vs_ess_kernel.db")

# %% load the kernel state if needed
# import dill
# dill.load_session("sss_vs_ess_kernel.db")

# %% drawing one histogram of the first coordinates with the target density
# values = [test_SSS[:, 0], test_ESS[:, 0], test_RWM[:, 0], test_pCN[:, 0]]
# count, bins, ignored = plt.hist(values, bins=20, density=True)
# plt.legend(labels.keys())
# plt.show()

# %% drawing separate histograms of the first coordinates with the target density
# draw_histogram_check(test_SSS[:, 0], labels["SSS"])
# draw_histogram_check(test_ESS[:, 0], labels["iESS"])
# draw_histogram_check(test_RWM[:, 0], labels["RWM"])
# draw_histogram_check(test_pCN[:, 0], labels["pCN"])
# h1 = plt.hist2d(test_SSS[:, 0], test_SSS[:, 1], bins=50)
# h2 = plt.hist2d(test_ESS[:, 0], test_ESS[:, 1], bins=50)
# h3 = plt.hist2d(test_RWM[:, 0], test_RWM[:, 1], bins=50)
# h4 = plt.hist2d(test_pCN[:, 0], test_pCN[:, 1], bins=50)


# %% simulating and drawing one path of the first coordinate of each algorithm
# sample_and_draw_path(random_SSS, labels["SSS"],  x0, 1000)
# sample_and_draw_path(random_ESS, labels["iESS"], x0, 1000)
# sample_and_draw_path(random_RWM, labels["RWM"],  x0, 1000)
# sample_and_draw_path(random_pCN, labels["pCN"],  x0, 1000)

# %% tune acceptance probability of RWM
# burn_in = 10**4
# N = 10**5
# d = 10
# x0 = np.zeros(d)
# for par in np.arange(0.79, 0.91, 0.05):
#     print(f"par = {par}")
#     random_RWM(par, N, burn_in, x0, test_func, print_time=False)
