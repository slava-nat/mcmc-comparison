# %% imports
import dill
import functools as ft
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

np.random.seed(1000)
# %% functions
def pi(x):
    """Calculate value of the density at the point x."""
    norm_x = np.linalg.norm(x)
    return np.exp(-norm_x - 0.5 * norm_x**2)

def pi2(x, y):
    """Basically pi(x) / pi(y)."""
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return np.exp(-norm_x + norm_y + 0.5*(-norm_x**2  + norm_y**2))

# global constant for acceptance_rates
# ACCEPTANCE_RATES = list()

def random_MH(size=1, burn_in=0, x0=[0], print_avg_acceptance_rate=False):
    """
    Perform the Maetropolis-Hastings Random Walk algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in.
    """
    d = len(x0)
    sampled_values = np.zeros((size, len(x0)))
    acceptance_rates = np.zeros((burn_in + size,))
    for i in range(-burn_in, size):
        # standard version:
        # x1 = x0 + np.random.normal(scale=0.3, size=d)
        # x0 = x1 if np.random.uniform() < pi2(x1, x0) else x0

        # version with the global constant:
        # ACCEPTANCE_RATES.append(pi2(x1, x0))
        # x0 = x1 if np.random.uniform() < ACCEPTANCE_RATES[-1] else x0

        # version with the local constant:
        x1 = x0 + np.random.normal(scale=0.3, size=d)
        acceptance_rates[i] = min(1, pi2(x1, x0))
        x0 = x1 if np.random.uniform() < acceptance_rates[i] else x0
        # save values after burn_in
        if i >= 0: sampled_values[i] = x0
    if print_avg_acceptance_rate:
        print("MH: average acceptance rate =", np.mean(acceptance_rates))
    return sampled_values

def runiform_ball(d, R=1):
    """
    Sample efficiently from a uniform distribution on a d-dimensional ball
    of radius R.
    """
    if R < 0: sys.exit("ERROR in runiform_ball: R must be nonnegative")
    x = np.random.normal(size=d)
    u = np.random.uniform()
    return R * u**(1 / d) * x / np.linalg.norm(x)

def random_SSS(size=1, burn_in=0, x0=[0]):
    """
    Perform the Simple Slice Sampler algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in.
    """
    sampled_values = np.zeros((size, len(x0)))
    for i in range(-burn_in, size):
        t  = np.random.uniform(0, pi(x0))
        x0 = runiform_ball(len(x0), -1 + np.sqrt(1 - 2*np.log(t)))
        # save values after burn_in
        if i >= 0: sampled_values[i] = x0
    return sampled_values

def random_two_segments(left_border, right_border, shift=np.pi):
    """
    Sample from a uniform distribution on a union of two swgments:
    [left_border, right_border] and [left_border + shift, right_border + shift]
    """
    if left_border > right_border:
        sys.exit("ERROR in random_two_segments: left_border must be smaller than right_border")
    x = np.random.uniform(left_border, right_border)
    return x + shift if np.random.binomial(1, 0.5) == 1 else x

def ellipse_point(x1, x2, angle):
    """Return a point on the ellipse generated by x1 and x2 with the angle a."""
    if not len(x1) == len(x2):
        sys.exit("ERROR in ellipse_point: lengths of x1 and x2 must be equal")
    return(x1 * np.cos(angle) + x2 * np.sin(angle))

def random_ESS(size=1, burn_in=0, x0=[0]):
    """
    Perform the Elliptical Slice Sampler algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in.
    """
    d = len(x0)
    sampled_values = np.zeros((size, d))
    for i in range(-burn_in, size):
        norm_x0 = np.linalg.norm(x0)
        t = np.random.uniform(0, np.exp(-norm_x0))
        w = np.random.normal(size=d)
        norm_w = np.linalg.norm(w)

        Ax = norm_x0**2 - norm_w**2
        Bx = 2 * np.sum(x0 * w)
        Cx = 2 * np.log(t)**2 - norm_x0**2 - norm_w**2

        phi = np.sign(Bx) * np.arccos(Ax / np.sqrt(Ax**2 + Bx**2))
        psi = np.arccos(min(1, Cx / np.sqrt(Ax**2 + Bx**2)))

        theta = random_two_segments((phi + psi) / 2, np.pi + (phi - psi) / 2)

        x0 = ellipse_point(x0, w, theta)
        # save values after burn_in
        if i >= 0: sampled_values[i] = x0
    return sampled_values

def rho(x):
    """Basically normalized pi for d = 1."""
    return (np.exp(-abs(x) - 0.5 * x**2)) / 1.31136

def draw_histogram_check(samples, title, bins=50, range=[-3, 3]):
    """Draw histogramm with rho over it."""
    count, bins, ignored = plt.hist(samples,
                                    bins=bins, density=True, range=range)
    plt.title(title)
    plt.plot(bins, rho(bins), color='r')
    plt.plot(bins, np.exp(-abs(bins)) / 2, color='b')
    plt.show()

def sample_and_draw_path(algorithm, title, x0, steps):
    """
    Simulate the algorithm given number of steps starting from x0 and
    draw the path of the first coordinate.
    """
    plt.title(title)
    plt.plot(range(steps), algorithm(steps, 0, x0)[:, 0])
    plt.show()

def get_correlations(samples, k_range=range(1, 11)):
    """
    Calculate correlations of "samples" for all k in "k_range".
    Returns a vector of length len("k_range")
    """
    N = len(samples)
    sum = np.sum(samples)
    denominator = np.sum((N * samples - sum)**2)
    length_k = len(k_range)
    corr = list()
    for k in k_range:
        nominator = 0
        for i in range(N - k):
            nominator += (N * samples[i] - sum) * (N * samples[i + k] - sum)
        corr.append(nominator / denominator)
    return corr

# %% set initial parameters
# dimension
d = 40
# starting vector of dimension "d"
x0 = np.zeros((d,))
# number of skipped iterations at the beginning
burn_in = 10**5
# number of iterations
N = 10**6
# set full names of the algorithms
labels = {"SSS": "Simple slice sampler",
          "ESS": "Elliptical slice sampler",
          "MH" : "Metropolis-Hastings"}

# %% test algorithms
test_SSS = random_SSS(N, burn_in, x0)
test_ESS = random_ESS(N, burn_in, x0)
test_MH  = random_MH (N, burn_in, x0, print_avg_acceptance_rate=True)

# %% save the kernel state
dill.dump_session("sss_vs_ess_kernel.db")

# %% load the kernel state if needed
import dill
dill.load_session("sss_vs_ess_kernel.db")

# %% drawing one histogram of the first coordinates with the target density
values = [test_SSS[:, 0], test_ESS[:, 0], test_MH[:, 0]]
count, bins, ignored = plt.hist(values, bins=50, density=True, label=labels.values())
plt.legend(loc="best")
plt.plot(bins, rho(bins), color='r')
plt.show()

# %% drawing separate histograms of the first coordinates with the target density
draw_histogram_check(test_SSS[:, 0], labels["SSS"])
draw_histogram_check(test_ESS[:, 0], labels["ESS"])
draw_histogram_check(test_MH [:, 0], labels["MH"])

# %% simulating and drawing one path of the first coordinate of each algorithm
sample_and_draw_path(random_SSS, labels["SSS"], x0, 1000)
sample_and_draw_path(random_ESS, labels["ESS"], x0, 1000)
sample_and_draw_path(random_MH,  labels["MH"],  x0, 1000)

# %% calculating autocorrelation of log(|x|)
# generate k-range evenly sapaced on a log scale
k_range = np.geomspace(start=1, stop=10**3, num=100, dtype=int)
# delete duplicates in k_range
k_range = sorted(list(set(k_range)))

corr_SSS = get_correlations(np.log(np.linalg.norm(test_SSS, axis=1)), k_range)
corr_ESS = get_correlations(np.log(np.linalg.norm(test_ESS, axis=1)), k_range)
corr_MH  = get_correlations(np.log(np.linalg.norm(test_MH,  axis=1)), k_range)

# %% plot correlations
plt.plot(k_range, corr_SSS)
plt.plot(k_range, corr_ESS)
plt.plot(k_range, corr_MH)
plt.title("40 dimensional autocorrelation fct for log(|x|)")
plt.xscale("log")
plt.legend(labels.values(), loc="best")
plt.savefig("Autocorrelation_norm.pdf")
plt.show()
