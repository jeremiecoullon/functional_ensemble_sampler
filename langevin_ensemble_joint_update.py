
import numpy as np
import multiprocessing as mp
from pathlib import Path
import time
import os

from forward_models import solve_Langevin
from BM_prior import samplePrior, x_range, dt, num_pt, get_KL_weights, inverseKL, sampleG, evects, evals, logPriorBM
from langevin_functions import true_path, le_obs, array_obs_points, log_post, sigma_obs

"Joint ensemble sampler"

# ============
# MCMC parameters
N = 20000000
thin_step = 200

omega = 0.15
M_trunc = 5
L = 100
a_prop = 2
# ============

# =====================
# TEST SAMPLER
# omega = 1
# N = 50000
# thin_step = 2
# L = 8
# from langevin_functions import log_prior_log_alpha, log_prior_log_sigma
# def log_post(W, log_alpha, log_sigma):
#     "Prior only: to test samplers"
#     log_p_alpha = log_prior_log_alpha(log_alpha)
#     log_p_sigma = log_prior_log_sigma(log_sigma)
#     return log_p_alpha + log_p_sigma
# =====================


if L < M_trunc+2:
    raise ValueError(f"Number of walkers must be >= {M_trunc+2} (d+2). ")


# Matrices to project on finite basis and CS:
project_M_lowfreq = evects[:, -M_trunc:] @ evects[:, -M_trunc:].T
project_M_highfreq = np.eye(num_pt) - project_M_lowfreq


Prec_lowrank = np.linalg.multi_dot([evects[:, -M_trunc:], np.diag(1/evals[-M_trunc:]), evects[:, -M_trunc:].T])

def low_rank_prior(xCurrent, xNew):
    "GP prior for finite dimensional subspace"
    return -0.5*(np.linalg.multi_dot([xNew, Prec_lowrank, xNew]) - np.linalg.multi_dot([xCurrent, Prec_lowrank, xCurrent]))


def ensemble_step(args):
    """
    Joint update for the low-dim subspace and the CS.

    Parameters:
    ----------
    args: list
        currentX, currentAlpha, currentSigma,
        otherX, otherAlpha, otherSigma,
        M_trunc, a_prop, omega,
        currentLogPost
    """
    currentX, currentAlpha, currentSigma, otherX, otherAlpha, otherSigma, M_trunc, a_prop, omega, currentLogPost = args

    # stretch move
    # w_j0 = get_KL_weights(otherX)[-M_trunc:]
    w_j0 = evects[:,-M_trunc:].T @ otherX # more efficient than using `get_KL_weights`
    w_k = get_KL_weights(currentX)
    w_k_start, w_k_end = w_k[:-M_trunc], w_k[-M_trunc:]
    Z = sampleG(a_prop)

    currentparm = np.concatenate([[currentAlpha, currentSigma], w_k_end])
    otherparm = np.concatenate([[otherAlpha, otherSigma], w_j0])

    alphaProp, sigmaProp, *wProp_end = otherparm*(1-Z) + Z*currentparm
    xPropJoint = inverseKL(np.concatenate([w_k_start, wProp_end]))

    xProp = (project_M_lowfreq + np.sqrt(1-omega**2)*project_M_highfreq)@xPropJoint + omega*project_M_highfreq@samplePrior()

    logPost_prop = log_post(W=xProp, log_alpha=alphaProp, log_sigma=sigmaProp)
    log_alpha = (M_trunc+2-1)*np.log(Z) + logPost_prop - currentLogPost + low_rank_prior(xCurrent=currentX, xNew=xProp) # + logPriorBM(xProp) - logPriorBM(currentX)
    if np.log(np.random.uniform(0,1)) < log_alpha:
        currentX = xProp[:]
        currentAlpha = alphaProp
        currentSigma = sigmaProp
        currentLogPost = logPost_prop
        acceptStretch = True
    else:
        acceptStretch = False
    return currentX, currentAlpha, currentSigma, currentLogPost, acceptStretch


# ================

def run_ensemble_sampler():
    N_thin = int(N/thin_step)

    samples = np.zeros((N_thin, 2, num_pt))
    samplesAlpha = np.zeros((N_thin, 2))
    samplesSigma = np.zeros((N_thin, 2))
    logPostList = np.zeros((N_thin, 2))

    currentAlpha = np.array([0+np.random.normal() for i in range(L)])
    currentSigma = np.array([0+np.random.normal() for i in range(L)])
    currentX = np.array([samplePrior() for e in range(L)])
    currentLogPost = np.zeros(L)

    for k in range(L):
        currentLogPost[k] = log_post(currentX[k,:], currentAlpha[k], currentSigma[k])

    logPostList[0, 0] = currentLogPost[0] # 0th walker
    logPostList[0, 1] = np.mean(currentLogPost) # average over all walkers
    num_accepts_joint = 0

    if 'global_storage' in os.environ:
        global_storage_path = os.environ['global_storage'] + "/"
    else:
        global_storage_path = ""
    dir_name = f"{global_storage_path}outputs/langevin_sampler_sine4/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_{a_prop}/joint_update"
    Path(f"{global_storage_path}outputs/langevin_sampler_sine4/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_{a_prop}").mkdir(exist_ok=True)
    Path(dir_name).mkdir(exist_ok=True)

    start_time = time.time()
    print(f"Running joint update ensemble sampler for {N} iterations. omega={omega}, M_trunc={M_trunc} and {L} walkers.\nProposal variance a={a_prop}\n")
    for i in range(N):
        mylist = list(range(L))
        np.random.shuffle(mylist)
        halfL = int(L / 2)
        S1, S2 = mylist[:halfL], mylist[halfL:L]

        Slist = [S1, S2]
        for idxS in [0,1]:
            S_current = Slist[idxS]
            S_other = Slist[idxS-1]
            S_arg_list = []
            for k in S_current:
                j0 = np.random.choice(S_other)
                arg_list = [currentX[k,:], currentAlpha[k], currentSigma[k],
                           currentX[j0,:], currentAlpha[j0], currentSigma[j0],
                           M_trunc, a_prop, omega,
                            currentLogPost[k]]
                S_arg_list.append(arg_list)

            results = pool.map(ensemble_step, S_arg_list)

            for k, (newX, newAlpha, newSigma, newLogPost, acceptJoint) in zip(S_current, results):
                currentX[k,:] = newX
                currentAlpha[k] = newAlpha
                currentSigma[k] = newSigma
                currentLogPost[k] = newLogPost
                num_accepts_joint += int(acceptJoint)
        if i%1000000==0:
            print(f"Iteration {i}/{N}")
            # save 0th walker and average over walkers
            np.savetxt(f"{dir_name}/ensemble_sampler-walker0-paths_L{L}.txt", samples[:, 0, :])
            np.savetxt(f"{dir_name}/ensemble_sampler-walker0-alpha_L{L}.txt", samplesAlpha[:, 0])
            np.savetxt(f"{dir_name}/ensemble_sampler-walker0-sigma_L{L}.txt", samplesSigma[:, 0])

            np.savetxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt", samples[:, 1, :])
            np.savetxt(f"{dir_name}/ensemble_sampler-average-alpha_L{L}.txt", samplesAlpha[:, 1])
            np.savetxt(f"{dir_name}/ensemble_sampler-average-sigma_L{L}.txt", samplesSigma[:, 1])
        # only save every N_thin iterations
        if i%thin_step==0:
            i_thin = int(i/thin_step)
            # save 0th walker
            samples[i_thin, 0, :] = currentX[0, :]
            logPostList[i_thin, 0] = currentLogPost[0]
            samplesSigma[i_thin, 0] = currentSigma[0]
            samplesAlpha[i_thin, 0] = currentAlpha[0]

            # save average over walkers
            samples[i_thin, 1, :] = np.mean(currentX, axis=0)
            logPostList[i_thin, 1] = np.mean(currentLogPost)
            samplesSigma[i_thin, 1] = np.mean(currentSigma)
            samplesAlpha[i_thin, 1] = np.mean(currentAlpha)
    print("Done")

    end_time = time.time()

    print(f"Running time {(end_time-start_time)/60:.2f}min")
    accept_rate_joint = num_accepts_joint / (N*L) * 100
    print(f"Acceptance rate: {accept_rate_joint:.1f}%")


    # save 0th walker and average over walkers
    np.savetxt(f"{dir_name}/ensemble_sampler-walker0-paths_L{L}.txt", samples[:, 0, :])
    np.savetxt(f"{dir_name}/ensemble_sampler-walker0-alpha_L{L}.txt", samplesAlpha[:, 0])
    np.savetxt(f"{dir_name}/ensemble_sampler-walker0-sigma_L{L}.txt", samplesSigma[:, 0])

    np.savetxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt", samples[:, 1, :])
    np.savetxt(f"{dir_name}/ensemble_sampler-average-alpha_L{L}.txt", samplesAlpha[:, 1])
    np.savetxt(f"{dir_name}/ensemble_sampler-average-sigma_L{L}.txt", samplesSigma[:, 1])
    with open(f"{dir_name}/ensemble_sampler_info.txt", 'w') as f:
        msg = f"""N = {N}\n\nthin_step={thin_step}\n\nL={L}\n\nomega={omega}\n\nM={M_trunc}, L={L}\n\nAcceptance rate: {accept_rate_joint:.1f}%"""
        f.write(msg)


if __name__ == '__main__':
    num_cores = mp.cpu_count()
    print(f"Num cores={num_cores}")
    pool = mp.Pool(num_cores)
    run_ensemble_sampler()
