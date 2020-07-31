
import numpy as np
from pathlib import Path
import time

from forward_models import solve_Langevin
from BM_prior import samplePrior, x_range, dt, num_pt, get_KL_weights, inverseKL, evects, evals, logPriorBM
from langevin_functions import true_path, le_obs, array_obs_points, log_post, sigma_obs, update_mean_cov

N = 3000000
thin_step = 100
M_trunc = 5
omega = 0.15

# ===============
# fit covariance to pre-run
print("Loading prerun and fitting proposal covariance...")
dir_name_prerun = "outputs/langevin_sampler/sigma-3_alpha-8/pCN_sampler/"


N_obs_adapt = 3000 # number of observations in prerun to fit mean and covariance. Note that the prerun is thinned by 100
samples_pCN_prerun = np.genfromtxt(f"{dir_name_prerun}/pCN_sampler-paths.txt")[:N_obs_adapt, :]
samples_alpha_pCN_prerun = np.genfromtxt(f"{dir_name_prerun}/pCN_sampler-alpha.txt")[:N_obs_adapt]
samples_sigma_pCN_prerun = np.genfromtxt(f"{dir_name_prerun}/pCN_sampler-sigma.txt")[:N_obs_adapt]

# Fit proposal cov to prerun
weights_samples_lowdim = np.array([get_KL_weights(e)[-M_trunc:] for e in samples_pCN_prerun[:]])
parm_array = np.array([samples_alpha_pCN_prerun[:], samples_sigma_pCN_prerun[:]])
parm_and_weights = np.concatenate([parm_array, weights_samples_lowdim.T])
# Fit proposal covariance for alpha and sigma
emp_cov = np.cov(parm_and_weights)
print("Done fitting proposal covariance.")
# ===============


# sin(t) only, with 5 observations. noise=0.2
# cov_parm_Hybrid = np.eye(2+M_trunc)
# cov_parm_Hybrid = delta_cov*np.eye(M_trunc+2) +  np.array([[  1.01790004,   0.4442157 ,   0.30058753,   0.88285695,
#          -1.23624875,  -1.22391285,   1.00444642],
#        [  0.4442157 ,   0.43803909,   0.1360243 ,   0.57271954,
#          -0.65955206,  -0.72719441,   0.4030441 ],
#        [  0.30058753,   0.1360243 ,   9.16967973,   0.40874836,
#          -0.35394315,  -1.70184415,   2.15310966],
#        [  0.88285695,   0.57271954,   0.40874836,  16.26344216,
#           0.73637355,   1.77241928,   0.9373708 ],
#        [ -1.23624875,  -0.65955206,  -0.35394315,   0.73637355,
#          30.91801477,   3.12280767,  -6.54185394],
#        [ -1.22391285,  -0.72719441,  -1.70184415,   1.77241928,
#           3.12280767,  61.73650684,  -3.07146941],
#        [  1.00444642,   0.4030441 ,   2.15310966,   0.9373708 ,
#          -6.54185394,  -3.07146941, 486.57869515]])
# rec_mean = np.array([  2.84504088,   0.70001065,  -0.31703484,   1.79229066,
#                  3.9293854 ,  -2.52407536, -24.51295593])
delta_cov = 1e-10
cov_parm_Hybrid = delta_cov*np.eye(M_trunc+2) + emp_cov
rec_mean = np.mean(parm_and_weights, axis=1)


# Matrices to project on finite basis and CS:
project_M_lowfreq = evects[:, -M_trunc:] @ evects[:, -M_trunc:].T
project_M_highfreq = np.eye(num_pt) - project_M_lowfreq


Prec_lowrank = np.linalg.multi_dot([evects[:, -M_trunc:], np.diag(1/evals[-M_trunc:]), evects[:, -M_trunc:].T])

def low_rank_prior(xCurrent, xNew):
    "GP prior for finite dimensional subspace"
    return -0.5*(np.linalg.multi_dot([xNew, Prec_lowrank, xNew]) - np.linalg.multi_dot([xCurrent, Prec_lowrank, xCurrent]))


N_thin = int(N/thin_step)

samplesHybrid = np.zeros((N_thin, num_pt))
samples_alphaHybrid = np.zeros(N_thin)
samples_sigmaHybrid = np.zeros(N_thin)

currentSampleHybrid = samplePrior()
current_alphaHybrid = 0
current_sigmaHybrid = 0

currentLogPost = log_post(W=currentSampleHybrid, log_alpha=current_alphaHybrid, log_sigma=current_sigmaHybrid)
log_post_list = np.zeros(N_thin)
log_post_list[0] = currentLogPost
num_accepts = 0

array_accepts = -1*np.ones(N-1)

dir_name = f"outputs/langevin_sampler/sigma-3_alpha-8/hybrid_sampler"
Path(dir_name).mkdir(exist_ok=True)

start_time = time.time()
print(f"Running Hybrid sampler for {N} iterations, omega={omega}, M_trunc={M_trunc}")
for i in range(1, N):
    # hybrid proposal
    w_current = get_KL_weights(currentSampleHybrid)
    w_c_start, w_c_end = w_current[:-M_trunc], w_current[-M_trunc:]
    alphaProp, sigmaProp, *w_c_prop = np.concatenate([[current_alphaHybrid, current_sigmaHybrid], w_c_end]) + omega*np.random.multivariate_normal(mean=np.zeros(2+M_trunc), cov=cov_parm_Hybrid)
    xPropHybrid = inverseKL(np.concatenate([w_c_start, w_c_prop]))

    # pCN proposal
    xProp = (project_M_lowfreq + np.sqrt(1-omega**2)*project_M_highfreq)@xPropHybrid + omega*project_M_highfreq@samplePrior()

    prop_LogPost = log_post(xProp, alphaProp, sigmaProp)
    AR_log_ratio = prop_LogPost - currentLogPost + low_rank_prior(xCurrent=currentSampleHybrid, xNew=xProp)
    if np.log(np.random.uniform(0,1)) < AR_log_ratio:
        num_accepts += 1
        array_accepts[i-1] = 1
        currentSampleHybrid = xProp
        current_alphaHybrid = alphaProp
        current_sigmaHybrid = sigmaProp
        currentLogPost = prop_LogPost
    else:
        array_accepts[i-1] = 0

    # update covariance and mean
    new_weights = evects[:,-M_trunc:].T @ currentSampleHybrid
    new_parm = [current_alphaHybrid, current_sigmaHybrid]
    new_parm_weight = np.concatenate([new_parm, new_weights])
    cov_hybrid_nodelta = cov_parm_Hybrid - delta_cov*np.eye(M_trunc+2) # remove delta*I before updating
    # recursive formula
    rec_mean, cov_hybrid_nodelta = update_mean_cov(rec_mean, cov_hybrid_nodelta, N_obs_adapt+i, new_parm_weight)
    cov_parm_Hybrid = cov_hybrid_nodelta + delta_cov*np.eye(M_trunc+2)

    if i%thin_step==0:
        i_thin = int(i/thin_step)
        log_post_list[i_thin] = currentLogPost
        samplesHybrid[i_thin] = currentSampleHybrid
        samples_alphaHybrid[i_thin] = current_alphaHybrid
        samples_sigmaHybrid[i_thin] = current_sigmaHybrid
    if i%50000==0:
        print(f"Iteration {i}/{N}")
        np.savetxt(f"{dir_name}/hybrid_sampler-paths.txt", samplesHybrid[:, :])
        np.savetxt(f"{dir_name}/hybrid_sampler-alpha.txt", samples_alphaHybrid[:])
        np.savetxt(f"{dir_name}/hybrid_sampler-sigma.txt", samples_sigmaHybrid[:])
        np.savetxt(f"{dir_name}/hybrid_sampler-array_accepts.txt", array_accepts)

accept_rate = num_accepts / N * 100
print("Done.")
print(f"Acceptance rate: {accept_rate:.2f}%")

np.savetxt(f"{dir_name}/hybrid_sampler-paths.txt", samplesHybrid[:, :])
np.savetxt(f"{dir_name}/hybrid_sampler-alpha.txt", samples_alphaHybrid[:])
np.savetxt(f"{dir_name}/hybrid_sampler-sigma.txt", samples_sigmaHybrid[:])
np.savetxt(f"{dir_name}/hybrid_sampler-array_accepts.txt", array_accepts)
with open(f"{dir_name}/hybrid_sampler_info.txt", 'w') as f:
    msg = f"""N = {N}\n\nthin_step={thin_step}\n\nomega={omega}\n\nM={M_trunc}\n\nAcceptance rate: {accept_rate:.1f}%"""
    f.write(msg)

end_time = time.time()
print(f"Running time: {(end_time-start_time)/60:.2f}min")
