import matplotlib.pyplot as plt
import numpy as np

from numba import njit

@njit
def first_coordinate(x):
    return x[0]

@njit
def random_MCMC(transition_func, size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate):
    """
    Perform the Markov Chain Monte Carlo algorithm with transition kernel
    defined by "transition_kernel" "burn_in" + "size" steps starting from x0.
    Function "transition kernel" must have only one argument: x, and return
    updated x and acceptance rate. If you implement an algorithm without
    acceptance-rejection method, simply return -1 as acceptance rate.

    Return a list of sampled vectors after burn_in. If "test_func" is given then
    return test_func(samples).
    """
    sampled_values = np.zeros(size)
    sum_acceptance_rates = 0
    # bollean for Metropolis-Hastings
    MH = True
    for i in range(-burn_in, size):
        x0, acceptance_rate = transition_func(x0)
        # check if the algorithm is using acceptance-rejection
        if acceptance_rate == -1: MH = False
        # save values after burn_in
        if i >= 0:
            sampled_values[i] = test_func(x0)
            if MH: sum_acceptance_rates += acceptance_rate
    if MH:
        print("average acceptance rate / average number of density evaluations =")
        print(sum_acceptance_rates / size)
    return sampled_values

def random_RWM(density_func, size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate):
    """
    Perform the Metropolis-Hastings Random Walk algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in. If "test_func" is given then return test_func(samples).
    """
    # define transition function.
    @njit
    def RWM_transition(x):
        d = len(x)
        x1 = x + np.random.normal(0, scale=2.5/np.sqrt(d), size=d)
        acceptance_rate = min(1, density_func(x1) / density_func(x))
        x = x1 if np.random.uniform(0, 1) < acceptance_rate else x
        return x, acceptance_rate

    return random_MCMC(RWM_transition, size, burn_in, x0, test_func)

def random_SSS(density_func, density_func_inv, size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate):
    """
    Perform the Simple Slice Sampler algorithm w.r.t. the density pi
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in. If "test_func" is given then return test_func(samples).
    """
    # define the necessary function
    @njit
    def runiform_disc(d, R=1, r=0):
        """
        Sample efficiently from a uniform distribution on a d-dimensional disc
        centered in zero D(R, r) = {x : r < |x| < R}.
        """
        x = np.random.normal(0, 1, size=d)
        # if r == 0 then sample more efficiently on a ball
        if r == 0:
            u = np.random.uniform(0, 1)
            return R * u**(1/d) * x / np.linalg.norm(x)
        # otherwise sample on a disc
        u = np.random.uniform(r**d, R**d)
        return u**(1/d) * x / np.linalg.norm(x)
    # define transition function. Return -1 on the second place (as acceptance rate)
    @njit
    def SSS_transition(x):
        t = np.random.uniform(0, density_func(x))
        R = density_func_inv(t)
        return runiform_disc(len(x), R, max(0, 2 - R)), -1

    return random_MCMC(SSS_transition, size, burn_in, x0, test_func)

@njit
def ellipse_point(x1, x2, angle):
    """Return a point on the ellipse generated by x1 and x2 with the angle a."""
    return(x1 * np.cos(angle) + x2 * np.sin(angle))

def random_ESS(density_func, size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate):
    """
    Perform the Elliptical Slice Sampler algorithm w.r.t. the "density_func"
    "burn_in" + "size" steps starting from "x0". Returns a list of sampled vectors
    after burn_in. If "test_func" is given then return test_func(samples).
    """
    # calculating the likelihood function
    @njit
    def likelihood_func(x):
        return density_func(x) * np.exp(0.5 * np.linalg.norm(x)**2)
    
    # define transition function. Return number of density calls on the second
    # place (as acceptance rate)
    @njit
    def ESS_transition(x):
        d = len(x)
        t = np.random.uniform(0, likelihood_func(x))
        w = np.random.normal(0, 1, size=d)

        x1 = np.zeros((d))
        theta_min = 0
        theta_max = 2 * np.pi
        number_of_density_calls = 0
        while 1:
            theta = np.random.uniform(theta_min, theta_max)
            x1 = ellipse_point(x, w, theta)
            number_of_density_calls += 1
            if likelihood_func(x1) >= t: break
            else:
                if theta < 0: theta_min = theta
                else:         theta_max = theta

        return x1, number_of_density_calls

    return random_MCMC(ESS_transition, size, burn_in, x0, test_func)

def random_pCN(density_func, size=1, burn_in=0, x0=np.zeros(1), test_func=first_coordinate):
    """
    Perform the Preconditioned Crank-Nicolson algorithm w.r.t. "density_func"
    "burn_in" + "size" steps starting from x0. Returns a list of sampled vectors
    after burn_in.
    """
    # calculating the likelihood function
    @njit
    def likelihood_func(x):
        return density_func(x) * np.exp(0.5 * np.linalg.norm(x)**2)
    
    # define transition kernel
    @njit
    def pCN_transition(x):
        norm_x = np.linalg.norm(x)
        w = np.random.normal(0, 1, size=len(x))

        x1 = ellipse_point(x, w, 1.5)
        acceptance_rate = min(1, likelihood_func(x1) / likelihood_func(x))
        x = x1 if np.random.uniform(0, 1) < acceptance_rate else x
        return x, acceptance_rate

    return random_MCMC(pCN_transition, size, burn_in, x0, test_func)

def draw_histogram_check(samples, title, bins=50, range=[-3, 3]):
    """Draw histogramm with rho over it."""
    count, bins, ignored = plt.hist(samples,
                                    bins=bins, density=True, range=range)
    plt.title(title)
    plt.show()

def sample_and_draw_path(algorithm, title, x0, steps):
    """
    Simulate the algorithm given number of steps starting from x0 and
    draw the path of the first coordinate.
    """
    plt.title(title)
    plt.plot(range(steps), algorithm(steps, 0, x0))
    plt.show()

@njit
def get_correlations(samples, k_range=range(1, 11)):
    """
    Calculate correlations of "samples" for all k in "k_range".
    Returns a vector of length len("k_range").
    """
    return [np.corrcoef(samples[:-k], samples[k:])[0, 1] for k in k_range]
