"""
In this file we check how Elliptical Slice Sampler performs on an example where
the target density is of "double banana" form.
"""

# %% imports
# standard library imports
import datetime
import numpy as np
import os
import time

# third party imports
import matplotlib.pyplot as plt
from numba import njit
from statsmodels.tsa import stattools

# local application imports
import mcmc

# %% set initial parameters
# dimension
d_range = [2]
# number of skipped iterations at the beginning
burn_in = 10**4
# number of iterations
N = 10**6
# the last k for calculating the autocorrelation function
k_max = 10**5
# starting vector of dimension "d"
x0 = np.zeros(2)
x0[1] = -1
# set parameters for the density
# sigma = 0.3
y_obs = 5
# define log of the PDF of double banana
@njit
def ln_pdf(x):
    return -abs(y_obs - np.log((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2))

# define test function
@njit
def test_func(x):
    return x

# %% calculating ACF for test_func
# set random seed
np.random.seed(1)
full_time = time.time()
acf = 0
start_time = time.time()
print("ESS started.")
samples = mcmc.random_ESS(ln_pdf, size=N, burn_in=burn_in, test_func=test_func, x0=x0)
print(f"ESS time: {datetime.timedelta(seconds=time.time() - start_time)}")
acf = stattools.acf(samples[:, 0], nlags=k_max, fft=True)[1:]

print(f"CPU time total = {datetime.timedelta(seconds=time.time() - full_time)}")

# %% create pics folder if needed
os.makedirs("pics", exist_ok=True)

# %% plot ACF
plt.plot(range(1, k_max + 1), acf)
plt.title("ACF")
plt.xscale("log")
plt.xlabel("Lag")
# plt.savefig(f"pics/ACF_d{d}.pdf")
plt.show()

# %% plot 2d-histogram
plt.hist2d(samples[:, 0], samples[:, 1], bins=100, range=[(-4, 4), (-4, 12)],
           rasterized=True)
plt.savefig("pics/double_banana.pdf")
plt.show()