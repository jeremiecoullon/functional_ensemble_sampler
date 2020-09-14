import numpy as np

from forward_models import solve_Langevin
from BM_prior import samplePrior, x_range, dt, num_pt, get_KL_weights, inverseKL, sampleG, evects, evals, logPriorBM


# true_path = np.sin(x_range)
# True path generated with alpha=5, sigma=0.5
# true_path = np.genfromtxt("data/langevin_problem/langevin_true_Xt.txt")

true_path = np.sin(4*x_range)
array_obs_points = np.arange(1, 10.5, 2)

# array_obs_points = np.arange(0.5, 10.5, 2)
obs_time_points = list(map(int, array_obs_points/dt - 1))
sigma_obs = 0.3

# le_obs = true_path[obs_time_points] + np.random.normal(loc=0, scale=sigma_obs, size=len(obs_time_points))
# np.savetxt("bimodal_observations.txt", le_obs)
# le_obs = np.genfromtxt("data/langevin_problem/bimodal_observations.txt")
le_obs = np.genfromtxt("data/langevin_problem/bimodal_observations_sin4.txt")


def loglikelihood(W, alpha, sigma):
    """
    Parameters
    W: ndarray
        BM path of size num_pt
    """
    X_t, P_t = solve_Langevin(W, alpha, sigma)
    predicted_values = X_t[obs_time_points]
    return -0.5 * (1/sigma_obs**2) * np.sum(np.square(le_obs - predicted_values))

sigma_hyperparam = 4
alpha_hyperparam = 12


def log_prior_log_sigma(log_sigma):
    """
    Prior for log-sigma. p(sigma)~Exponential(1)

    Parameters
    ----------
    log_sigma: log-sigma
    """
    return log_sigma - np.exp(log_sigma)/sigma_hyperparam

def log_prior_log_alpha(log_alpha):
    """
    Prior for log-alpha. p(alpha)~Exponential(1)

    Parameters
    ----------
    log_alpha: log-alpha
    """
    return log_alpha - np.exp(log_alpha)/alpha_hyperparam



def log_post(W, log_alpha, log_sigma):
    "Log-posterior"
    log_p_alpha = log_prior_log_alpha(log_alpha)
    log_p_sigma = log_prior_log_sigma(log_sigma)

    return loglikelihood(W, np.exp(log_alpha), np.exp(log_sigma)) + log_p_alpha + log_p_sigma


def update_mean_cov(rec_mean, rec_cov, N, x):
    """
    Parameters
    ----------
    rec_mean, rec_cov: ndarray
        Recursive estimate of mean and cov
    N: int
        Total number of samples used in update (including sample x below)
    x: ndarray
        Sample x to update mean and cov with

    Returns
    -------
    rec_mean, rec_cov: ndarray
        Recursive estimate of mean and cov
    """
    rec_cov = (N-2)/(N-1)*rec_cov + (1/N)*np.outer(x-rec_mean, x-rec_mean)
    rec_mean = rec_mean * (N-1)/(N) + x/N
    return rec_mean, rec_cov
