"""
In this module you can find functions for generating samples by using different
Markov Chain Monte Carlo methods (MCMC), such as:
    * Random Walk Metropolis,
    * Simple Slice Sampler,
    * Elliptical Slice Sampler,
    * Preconditioned Crank-Nicolson.

Dependencies: numba
"""

# standard library imports
import math
import matplotlib.pyplot as plt
import numpy as np

# third party imports
from numba import njit

@njit
def first_coordinate(x):
    """ Return the first coordinate of the given vector. """
    return np.array([x[0]])

@njit
def random_MCMC(transition_func, size=1, burn_in=0, x0=np.zeros(1),
                test_func=first_coordinate):
    """
    Perform a Markov Chain Monte Carlo method with transition kernel
    "transition_func". Returns a list of sampled vectors after burn_in.
    If "test_func" is given then return test_func(samples).

    Parameters
    ----------
    transition_func : function(ndarray) returning ndarray, float
        Transition kernel samples next state given the current one.
        Must return a tuple (next_state, par), where `par` is the
        parameter, for which the average value will be printed.
        You can collect e.g. number of density calls or
        acceptance ratio. If not needed simply return par=-1.
        For better performance you should wrap your transition
        function using the wrapper @njit from the package "numba".
    size : int, optional
        Size of the sample, by default 1
    burn_in : int, optional
        Number of iterations to be skipped at the beginning, by default 0.
    x0 : ndarray, optional
        Starting vector of the algorithm, by default (0).
    test_func : function(ndarray), returning ndarray, optional
        Function of the samples to be saved, by default save only
        the first coordinate. The function must always return an ndarray.

    Returns
    -------
    sampled_values: ndarray
        Return a list of sampled vectors after burn_in. If "test_func" is given then
    return test_func(samples).

    Examples
    -------
    Random-Walk-Metropolis
    >>> from numba import njit
        @njit
        def RWM_transition(x):
            x1 = x + np.random.normal(0, 1, size=len(x))
            acceptance_rate = min(1, pdf(x1) / pdf(x))
            x = x1 if np.random.uniform(0, 1) < acceptance_rate else x
            return x, acceptance_rate
        sample = random_MCMC(RWM_transition, size=100, x0=np.zeros(2))
    """
    sampled_values = np.zeros((size, len(test_func(x0))))
    # sampled_values = np.zeros(size)
    sum_acceptance_rates = 0
    # boolean for Metropolis-Hastings
    MH = True
    for i in range(-burn_in, size):
        x0, acceptance_rate = transition_func(x0)
        # check if the algorithm is using acceptance-rejection
        if acceptance_rate == -1: MH = False
        # save values after burn_in
        if i >= 0:
            sampled_values[i] = test_func(x0)
            if MH: sum_acceptance_rates += acceptance_rate
    # if MH:
    #     print("average acceptance rate / average number of density evaluations =")
    #     print(sum_acceptance_rates / size)
    return sampled_values

def random_RWM(ln_pdf, sd=1, **kwargs):
    """
    Perform the Metropolis-Hastings Random Walk algorithm and
    sample w.r.t. the given density dunction.

    Parameters
    ----------
    ln_pdf : fucntion(ndarray), return float
        Function calculating the logarithm of the value of density function.
        For better performance you should wrap your function using the
        wrapper @njit from the package "numba".
    sd : float, optional
        Standard deviation determing each step of the algorithm, by default 1.
    **kwargs: keyword arguments
        Standard parameters, such as `size`, `burn_in` of Markov Chain Monte
        Carlo algorithm to pass to the function `random_MCMC`. For details
        see the description of `random_MCMC`.
    
    Returns
    -------
    sampled_values: ndarray
        Return a list of sampled vectors.
    
    Examples
    ----------
    >>> random_RWM(lambda x: numpy.exp(-x**2), size=100, burn_in=10)
    """
    @njit
    def RWM_transition(x):
        """
        Transition function of RWM. Returns the new state and the acceptance
        rate.
        """
        d = len(x)
        # determine the proposal
        x1 = x + np.random.normal(0, sd, size=d)
        # determine the acceptance probability
        acceptance_rate = min(1, np.exp(ln_pdf(x1) - ln_pdf(x)))
        # acceptance-rejection
        x = x1 if np.random.uniform(0, 1) < acceptance_rate else x
        return x, acceptance_rate

    return random_MCMC(RWM_transition, **kwargs)

def random_SSS(ln_pdf, runiform_levelset, **kwargs):
    """
    Perform the Simple Slice Sampler algorithm w.r.t. the given density
    function.

    Parameters
    ----------
    ln_pdf : fucntion(ndarray), return float
        Function calculating the logarithm of the value of density function.
        For better performance you should wrap your function using the
        wrapper @njit from the package "numba".
    runiform_levelset : fucntion(float), return ndarray
        Function sampling uniform distribution on the given level set.
        For better performance you should wrap your function using the
        wrapper @njit from the package "numba".
    **kwargs: keyword arguments
        Standard parameters, such as `size`, `burn_in` of Markov Chain Monte
        Carlo algorithm to pass to the function `random_MCMC`. For details
        see the description of `random_MCMC`.

    Returns
    ----------
    sampled_values: ndarray
        Return a list of sampled vectors.
    
    Examples
    ----------
    >>> random_SSS(lambda x: -x**2,
                   lambda t: numpy.sqrt(-numpy.log(t)) * numpy.random.uniform(-1, 1)
                   size=100, burn_in=10)
    """
    @njit
    def SSS_transition(x):
        """ Transition function of SSS. Returns the new state and -1. """
        # determine the level t
        t = np.random.uniform(0, np.exp(ln_pdf(x)))
        # sample new state uniformly on the level set
        return runiform_levelset(len(x), t), -1

    return random_MCMC(SSS_transition, **kwargs)

@njit
def ellipse_point(x1, x2, angle):
    """ Return a point on the ellipse: x1 * cos(angle) + x2 * sin(angle). """
    return x1 * np.cos(angle) + x2 * np.sin(angle)

def random_ESS(ln_pdf, sd=1, **kwargs):
    """
    Perform the Elliptical slice sampler algorithm and
    sample w.r.t. the given density dunction.

    Parameters
    ----------
    ln_pdf : fucntion(ndarray), return float
        Function calculating the logarithm of the value of density function.
        For better performance you should wrap your function using the
        wrapper @njit from the package "numba".
    sd : float, optional
        Standard deviation of ellipse determing normal vector, by default 1.
    **kwargs: keyword arguments
        Standard parameters, such as `size`, `burn_in` of Markov Chain Monte
        Carlo algorithm to pass to the function `random_MCMC`. For details
        see the description of `random_MCMC`.
    
    Returns
    -------
    sampled_values: ndarray
        Return a list of sampled vectors.
    
    Examples
    ----------
    >>> random_ESS(lambda x: numpy.exp(-x**2), size=100, burn_in=10)
    """
    # calculating the log-likelihood function corrected by the prior
    @njit
    def log_likelihood_func(x):
        return ln_pdf(x) + 0.5/sd**2 * np.linalg.norm(x)**2
    
    @njit
    def ESS_transition(x):
        """
        Transition function of ESS. Returns the new state and number of density
        calls on the second place (as acceptance rate). If
        """
        d = len(x)
        # determine the level t and its log
        u = np.random.uniform(0, 1)
        log_t = np.log(u) + log_likelihood_func(x)
        # determine the ellipse
        w = np.random.normal(0, sd, size=d)
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

    return random_MCMC(ESS_transition, **kwargs)

def random_pCN(ln_pdf, sd=1, angle_par=0.25, **kwargs):
    """
    Perform the Preconditioned Crank-Nicolson algorithm and
    sample w.r.t. the given density dunction.

    Parameters
    ----------
    ln_pdf : fucntion(ndarray), return float
        Function calculating the logarithm of the value of density function.
        For better performance you should wrap your function using the
        wrapper @njit from the package "numba".
    sd : float, optional
        Standard deviation of ellipse determing normal vector, by default 1.
    angle_par : float
        Parameter between 0 and 2*pi determing the proposal by
        x * cos(angle_par) + w * cos(angle_par),
        where x is a current state, w is a vectrored sampled from the normal
        distribution with mean 0 and covariance matrix `cov_matrix`.
        Tune this parameter to have an average acceptance proability
        around 0.25.
    **kwargs: keyword arguments
        Standard parameters, such as `size`, `burn_in` of Markov Chain Monte
        Carlo algorithm to pass to the function `random_MCMC`. For details
        see the description of `random_MCMC`.
    
    Returns
    -------
    sampled_values: ndarray
        Return a list of sampled vectors.
    
    Examples
    ----------
    >>> random_pCN(lambda x: numpy.exp(-x**2), size=100, burn_in=10, angle_par=0.5)
    """
    # calculating the log-likelihood function corrected by the prior
    @njit
    def log_likelihood_func(x):
        return ln_pdf(x) + 0.5/sd**2 * np.linalg.norm(x)**2
    
    @njit
    def pCN_transition(x):
        """
        Transition function of pCN. Returns the new state and the acceptance
        rate.
        """
        d = len(x)
        # determine the ellipse
        w = np.random.normal(0, sd, size=d)
        # determine the proposal
        x1 = ellipse_point(x, w, angle_par)
        # determine the acceptance probability
        acceptance_rate = min(1, np.exp(log_likelihood_func(x1) - log_likelihood_func(x)))
        # acceptance-rejection
        x = x1 if np.random.uniform(0, 1) < acceptance_rate else x
        return x, acceptance_rate

    return random_MCMC(pCN_transition, **kwargs)

def sample_and_draw_path(algorithm, title, ploted_steps=100, **alg_kwargs):
    """
    Simulate the algorithm given number of steps starting from x0 and
    draw the path of the first coordinate.
    """
    samples = algorithm(**alg_kwargs)
    steps = len(samples)
    samples_thru = samples[::int(steps / ploted_steps)]
    steps_thru = range(steps)[::int(steps / ploted_steps)]
    plt.subplot(1, 2, 1)
    plt.title(title)
    plt.plot(steps_thru, samples_thru)
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=50, density=True, orientation="horizontal")
    plt.show()