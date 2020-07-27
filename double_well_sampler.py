
import numpy as np
import time
import multiprocessing as mp
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path

from BM_prior import samplePrior, get_KL_weights, inverseKL, evects, evals, sampleG, dt, num_pt
from forward_models import solve_DW

"""
Double well sampler
Data: 20 observations with obs_noise = 0.1
"""


N = 10000
thin_step = 100

omega = 0.24
M_trunc = 10
L = 20
a_prop = 1.5

# ================
# ================


true_BM_sample = np.genfromtxt("data/conditioned_diff_true_path.txt")
true_path = solve_DW(true_BM_sample)

# 20 observations
array_obs_points = np.arange(0.5, 10.5, 0.5)

obs_time_points = list(map(int, array_obs_points/dt - 1))
sigma_obs = 0.1
# le_obs = true_path[obs_time_points] + np.random.normal(loc=0, scale=sigma_obs, size=len(obs_time_points))
le_obs = np.genfromtxt("data/conditioned_diff_obs.txt")


def loglikelihood(W):
    """
    Parameters
    W: ndarray
        BM path of size num_pt
    """
    P_t = solve_DW(W)
    predicted_values = P_t[obs_time_points]
    return -0.5 * (1/sigma_obs**2) * np.sum(np.square(le_obs - predicted_values))



Prec_lowrank = np.linalg.multi_dot([evects[:, -M_trunc:], np.diag(1/evals[-M_trunc:]), evects[:, -M_trunc:].T])

def low_rank_prior(xCurrent, xNew):
    "GP prior for finite dimensional subspace"
    return -0.5*(np.linalg.multi_dot([xNew, Prec_lowrank, xNew]) - np.linalg.multi_dot([xCurrent, Prec_lowrank, xCurrent]))


if L < M_trunc+1:
    raise ValueError(f"Number of walkers must be >= {M_trunc+1} (d+1). ")

# Matrices to project on finite basis and CS:
project_M_lowfreq = evects[:, -M_trunc:] @ evects[:, -M_trunc:].T
project_M_highfreq = np.eye(num_pt) - project_M_lowfreq


def ensemble_step_DW(args):
    """
    stetch move and pCN move for the ensemble sampler

    Parameters:
    ----------
    args: list
        currentX, otherX,
        M_trunc, a_prop, omega,
        currentLogPost
    """
    currentX, otherX, M_trunc, a_prop, omega, currentLogPost = args

    # stretch move
    w_j0 = get_KL_weights(otherX)[-M_trunc:]
    w_k = get_KL_weights(currentX)
    w_k_start, w_k_end = w_k[:-M_trunc], w_k[-M_trunc:]
    Z = sampleG(a_prop)
    wProp_end = w_j0*(1-Z) + Z*w_k_end
    xProp = inverseKL(np.concatenate([w_k_start, wProp_end]))

    logPost_prop = loglikelihood(W=xProp)
    log_alpha = (M_trunc-1)*np.log(Z) + logPost_prop - currentLogPost + low_rank_prior(xCurrent=currentX, xNew=xProp)
    if np.log(np.random.uniform(0,1)) < log_alpha:
        currentX = xProp[:]
        currentLogPost = logPost_prop
        acceptStretch = True
    else:
        acceptStretch = False

    # pCN move
    xProp = (project_M_lowfreq + np.sqrt(1-omega**2)*project_M_highfreq)@currentX + omega*project_M_highfreq@samplePrior()
    newLogPost = loglikelihood(W=xProp)
    log_alpha = newLogPost - currentLogPost
    if np.log(np.random.uniform(0,1)) < log_alpha:
        return xProp, newLogPost, acceptStretch, True
    else:
        return currentX, currentLogPost, acceptStretch, False



# ================
# ================
num_cores = mp.cpu_count()
pool = mp.Pool(7)


N_thin = int(N/thin_step)

# initialise MCMC:
# samples N_thin samples (ie: thin samples)
# Save 2 MCMC chains: 1. one of the walkers and 2. the average over walkers
samples = np.zeros((N_thin, 2, num_pt))
logPostList = np.zeros((N_thin, 2))


currentX = np.array([np.sqrt(1-0.1**2)*true_BM_sample + 0.1*samplePrior() for e in range(L)])
currentLogPost = np.zeros(L)


for k in range(L):
    currentLogPost[k] = loglikelihood(currentX[k,:])

logPostList[0, 0] = currentLogPost[0] # 0th walker
logPostList[0, 1] = np.mean(currentLogPost) # average over all walkers
num_accepts_AIES = 0
num_accepts_PCN = 0


dir_name = f"outputs/double_well_sampler/ensemble_sampler/L_{L}-a_{a_prop}"
Path(dir_name).mkdir(exist_ok=True)



start_time = time.time()
print(f"Running function space AIES for {N} iterations. M_trunc={M_trunc} and {L} walkers.\nProposal variance a={a_prop}\n")
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
            arg_list = [currentX[k,:], currentX[j0,:],
                       M_trunc, a_prop, omega,
                        currentLogPost[k]]
            S_arg_list.append(arg_list)

        results = pool.map(ensemble_step_DW, S_arg_list)

        for k, (newX, newLogPost, acceptStretch, acceptpCN) in zip(S_current, results):
            currentX[k,:] = newX
            currentLogPost[k] = newLogPost
            num_accepts_AIES += int(acceptStretch)
            num_accepts_PCN += int(acceptpCN)

    # only save every N_thin iterations
    if i%thin_step==0:
        i_thin = int(i/thin_step)
        # save 0th walker
        samples[i_thin, 0, :] = currentX[0, :]
        logPostList[i_thin, 0] = currentLogPost[0]

        # save average over walkers
        samples[i_thin, 1, :] = np.mean(currentX, axis=0)
        logPostList[i_thin, 1] = np.mean(currentLogPost)

    if i%2000 == 0:
        print(f"Iteration {i}/{N}")
        # save 0th walker and average over walkers
        np.savetxt(f"{dir_name}/ensemble_sampler-walker0-paths_L{L}.txt", samples[:, 0, :])
        np.savetxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt", samples[:, 1, :])



end_time = time.time()
running_time = (end_time-start_time)/60
print(f"Running time: {running_time:.2f} min")

accept_rate_AIES = num_accepts_AIES / (N*L) * 100
accept_rate_PCN = num_accepts_PCN / (N*L) * 100
print(f"Acceptance rate for AIES: {accept_rate_AIES:.1f}%")
print(f"Acceptance rate for pCN: {accept_rate_PCN:.1f}%")




# save 0th walker and average over walkers
np.savetxt(f"{dir_name}/ensemble_sampler-walker0-paths_L{L}.txt", samples[:, 0, :])
np.savetxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt", samples[:, 1, :])
with open(f"{dir_name}/ensemble_sampler_info.txt", 'w') as f:
    msg = f"""N = {N}\n\nthin_step={thin_step}\n\nL={L}\n\nomega={omega}\n\nM={M_trunc}, L={L}\n\nAcceptance rates:\nensemble: {accept_rate_AIES:.1f}%\npCN: {accept_rate_PCN:.1f}%"""
    f.write(msg)
