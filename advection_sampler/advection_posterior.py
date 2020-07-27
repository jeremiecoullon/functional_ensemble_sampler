import numpy as np
from .advectionequation import log_lik_advection, gen_advection_data

prior_mean = 100
l, sigma_prior= 1, 130
# Likelihood noise
loss_sd = 0.5

def cov_exp(x,y):
    return sigma_prior*np.exp(-0.5*(1/l)*(x-y)**2)
x_min, x_max = 0, 10

num_pt = 200
IC_prior_mean = np.ones(num_pt)*prior_mean
xs = np.linspace(x_min, x_max, num_pt)
# build covariance matrix
XX, YY = np.meshgrid(xs, xs)
cov_prior = cov_exp(XX, YY) + np.eye(num_pt)*1e-4
cov_chol = np.linalg.cholesky(cov_prior)
precision_mat = np.linalg.inv(cov_prior)

def samplePrior():
    return np.dot(cov_chol, np.random.normal(size=num_pt))

# ===============
# generate true_IC
true_IC = samplePrior() + IC_prior_mean

# Use saved true_IC & data array
true_IC = np.genfromtxt("data/thesis_Advection_true_IC.csv")
data_array = np.genfromtxt("data/advection_data_array_sigma05.csv")

# Generate advection data
true_u = 0.5

num_data = 6
# list_locations = [(k,v) for k, v in zip(np.random.uniform(2, 10, size=num_data), np.random.uniform(0,2, size=num_data))]
list_locations = [(6.229025183588771, 1.046262032840266),
 (3.9796996427900364, 1.8577904239602372),
 (9.10480397079595, 1.249327847984541),
 (9.971123877107935, 1.9549391760550934),
 (7.098389475198764, 1.1756014323922388),
 (8.53772852322852, 0.03540495948154798)]


def uniform_log_prob(theta, lower, upper):
    if lower < theta < upper:
        return np.log(1/(upper - lower))
    else:
        return -np.inf

def log_prior_u(u):
    lower = 0
    upper = 1.4
    return uniform_log_prob(theta=u, lower=lower, upper=upper)


def logLik(u, IC):
    "log-posterior for both u and IC"
    if not (0<u<1.4):
        return -9999999999
    log_lik = log_lik_advection(u=u, IC=IC, data_array=data_array,
                                x_min=x_min, x_max=x_max, error_model='gaussian', loss_sd=loss_sd)
    return log_lik + log_prior_u(u)
