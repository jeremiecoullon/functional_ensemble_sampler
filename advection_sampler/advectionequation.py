# -*- coding: utf-8 -*-

import numpy as np

def solve_advection(u, IC, x, t, x_min=0, x_max=10):
    """
    Returns density and flow at (x,t) for a given wavespeed and initial condition

    Parameters
    ----------
    u: float
        Wavespeed of advection equation
    IC: ndarray
        Array of initial condition. Space between 0 and 10
    x: float
        Location to calculate density and flow for
    t: float
        Time to calculate density and flow for
    x_min, x_max: float
        Minimum and maximum value of x

    Returns
    -------
    den: float
        Density at (x,t)
    flow: float
        Flow at (x,t)
    """
    # x point in IC corresponding to solution
    IC_x = x - u*t
    if not x_min<=IC_x<=x_max:
        raise ValueError("IC is within [{0}, {1}]. Requested solution is at x = {2:.1f}".format(x_min, x_max, IC_x))
    # find index of IC that is the closest point inkk space to the required point
    idx_IC = np.abs(np.linspace(x_min, x_max, len(IC)) - IC_x).argmin()
    den, flow = IC[idx_IC], IC[idx_IC]*u
    return den, flow


def gen_advection_data(u, IC, list_locations, poisson_error=False, gaussian_error=False, sd=1, x_min=0, x_max=10):
    """
    Returns an array of flow data and detector locations: (x, t, flow)

    Parameters
    ----------
    u: float
        Wavespeed of advection equation
    IC: ndarray
        Array of initial condition. Space between 0 and 10
    list_locations: list
        List of tuples (x,t)
    poisson_error: Bool
        Whether or noot to add poisson error to observations

    Returns
    -------
    data_array: ndarray
        Array of data of size (N,3) (for N number of data points). Format: x, t, flow
    """
    list_data = []
    for x,t in list_locations:
        _, flow = solve_advection(u=u, IC=IC, x=x, t=t, x_min=x_min, x_max=x_max)
        if poisson_error==True:
            flow = np.random.poisson(flow)
        if gaussian_error==True:
            flow = np.random.normal(loc=flow, scale=sd)
        list_data.append([x,t,flow])
    return np.array(list_data)

def log_lik_advection(u, IC, data_array, x_min=0, x_max=10, error_model='gaussian', loss_sd=1):
    """
    Returns log-likelihood of u and IC for the advection equation using a Poisson error model

    Parameters
    ----------
    u: float
        Wavespeed of advection equation
    IC: ndarray
        Array of initial condition. Space between 0 and 10
    data_array: ndarray
        Array of data of size (N,3) (for N number of data points). Format: x, t, flow
    error_model: str
        Either 'poisson' or 'gaussian'
    loss_sd: float
        Standard deviation of gaussian error

    Returns
    -------
    log_lik: float
        Un-normalised log likelihood under the Poisson model
    """
    log_lik_list = []
    for (x, t, obs_flow) in data_array:
        _, pred_flow = solve_advection(u=u, IC=IC, x=x, t=t, x_min=x_min, x_max=x_max)
        if error_model == 'poisson':
            log_lik_detector = obs_flow*np.log(pred_flow) - pred_flow
        elif error_model == 'gaussian':
            log_lik_detector = -0.5*(1/loss_sd**2) * (obs_flow - pred_flow)**2
        else:
            raise ValueError("error_model should be either 'poisson' or 'gaussian'")
        log_lik_list.append(log_lik_detector)
    log_lik = np.sum(log_lik_list)
    return log_lik
