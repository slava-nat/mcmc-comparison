"""
In this fie we investigate Elliptical Slice Sampler on an example where the
target density is an indicator function of a d-dimensional cube [0,1]^d. This
kind of density is not lower semicontinous. Our hypothesis is that lower
semicontinuity is the minimum requirement for Elliptical Slice Sampler to be
reversible. The challenging starting point in this example is x0=0, because then
with positive probability one stays at this point and cannot escape it.
"""

# %% imports
# standard library imports
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# third party imports
from numba import njit

# local application imports
import mcmc

# %% set initial parameters
# dimension
dimensions = range(1, 9)
# number of times to perform one step of ESS
N = 10**4

# define log of the indicator function of the d-dimensional cube [0, 1]^d
@njit
def ln_pdf(x):
    """Calculate log value of the density at the point x."""
    for i in range(len(x)):
        if x[i] < 0 or x[i] > 1:
            return np.log(0)
    return 0

# define test function for storing the samples
@njit
def norm(x):
    return np.array([np.linalg.norm(x)])

# %% define neccessary functions
# define function returning point on th ellipse
@njit
def ellipse_point(x1, x2, angle):
    """Return a point on the ellipse x1 * cos(angle) + x2 * sin(angle)"""
    return x1 * np.cos(angle) + x2 * np.sin(angle)

# define log likelihood function
@njit
def log_likelihood_func(x):
    return ln_pdf(x) + 0.5 * np.linalg.norm(x)**2

# define transition function. Return number of density calls on the second
# place (as acceptance rate)
@njit
def ESS_transition(x):
    d = len(x)
    # determine the level t and its log
    u = np.random.uniform(0, 1)
    log_t = np.log(u) + log_likelihood_func(x)
    # determine the ellipse
    w = np.random.normal(0, 1, size=d)
    # set initial bracket
    theta = np.random.uniform(0, 2 * np.pi)
    theta_min = theta - 2 * np.pi
    theta_max = theta
    # initial value of the next state
    x1 = np.zeros(d)

    number_of_density_calls = 0
    # slice sampling loop
    while True:
        # compute proposal
        x1 = ellipse_point(x, w, theta)

        number_of_density_calls += 1
            
        if log_likelihood_func(x1) > log_t:
            # x1 is on the slice
            break
        else:
            # shrink the angle bracket
            if theta < 0: theta_min = theta
            else:         theta_max = theta
        # sample new angle
        theta = np.random.uniform(theta_min, theta_max)

    return x1, number_of_density_calls

# %% perform one step of ESS "N" times for each dimension and count escapes
escapes = {}
start_time = time.time()
for d in dimensions:
    trials = np.zeros(N)
    escapes[d] = 0
    # starting point
    x0 = np.zeros(d)
    for i in range(N):
        trials[i] = mcmc.random_MCMC(ESS_transition,
                                    size=1,
                                    x0=x0,
                                    test_func=norm)[:, 0]
        if trials[i] > 0.001:
            escapes[d] += 1
total_time = datetime.timedelta(seconds=time.time() - start_time)
print(f"CPU time total = {total_time}")

# %% calculate escape ratios
escape_ratios = [escapes[d]/N for d in dimensions]

# %% create pics folder if needed
os.makedirs("pics", exist_ok=True)

# %% plot escape ratio along with the estimate
estimate = [2**(1-d) for d in dimensions]
plt.plot(dimensions, escape_ratios, '-s')
plt.plot(dimensions, estimate, '--o')
plt.yscale("log")
plt.title("Escape ratios from starting point " + r"$x_0=0$" + "\n" +
          f"of elliptical slice sampler (with {N} trials)\n" +
          r"for the denisty $\rho(x) = \mathbf{1}_{[0,1]^d}(x)$")
plt.xlabel("Dimension")
plt.legend(["escape ratios of ESS", r"estimate: $2^{1-d}$"])
plt.savefig("pics/escape_ratio.pdf", bbox_inches="tight")
plt.show()