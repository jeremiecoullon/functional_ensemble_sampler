
import numpy as np
from pathlib import Path
import time

from forward_models import solve_Langevin
from BM_prior import samplePrior, x_range, dt, num_pt, get_KL_weights, inverseKL, evects, evals, logPriorBM
from langevin_functions import true_path, le_obs, array_obs_points, log_post, sigma_obs, update_mean_cov

N = 5000000
thin_step = 200
M_trunc = 5
omega = 0.15

# ===============
# fit covariance to pre-run
print("Loading prerun and fitting proposal covariance...")
dir_name_prerun = "outputs/langevin_sampler_sine4/sigma-4_alpha-12/pCN_sampler/"


N_obs_adapt = 5000 # number of observations in prerun to fit mean and covariance. Note that the prerun is thinned by 100
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
# alpha_variance_array = np.zeros(N_thin)
cov_diag_array = np.zeros((N_thin, M_trunc+2))

currentSampleHybrid = samplePrior()
current_alphaHybrid = 0
current_sigmaHybrid = 0

currentLogPost = log_post(W=currentSampleHybrid, log_alpha=current_alphaHybrid, log_sigma=current_sigmaHybrid)
log_post_list = np.zeros(N_thin)
log_post_list[0] = currentLogPost
# alpha_variance_array[0] = cov_parm_Hybrid[0,0] - delta_cov
cov_diag_array[0,:] = np.diag(cov_parm_Hybrid) - np.ones(M_trunc+2)*delta_cov
num_accepts = 0

array_accepts = -1*np.ones(N-1)

# dir_name = f"outputs/langevin_sampler/sigma-3_alpha-8/hybrid_sampler"
dir_name = f"outputs/langevin_sampler_sine4/sigma-4_alpha-12/hybrid_sampler"
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
        # alpha_variance_array[i_thin] = cov_parm_Hybrid[0,0] - delta_cov
        cov_diag_array[i_thin,:] = np.diag(cov_parm_Hybrid) - np.ones(M_trunc+2)*delta_cov
    if i%50000==0:
        print(f"Iteration {i}/{N}")
        np.savetxt(f"{dir_name}/hybrid_sampler-paths.txt", samplesHybrid[:, :])
        np.savetxt(f"{dir_name}/hybrid_sampler-alpha.txt", samples_alphaHybrid[:])
        np.savetxt(f"{dir_name}/hybrid_sampler-sigma.txt", samples_sigmaHybrid[:])
        np.savetxt(f"{dir_name}/hybrid_sampler-array_accepts.txt", array_accepts)
        # np.savetxt(f"{dir_name}/hybrid_sampler-alpha_variance_array.txt", alpha_variance_array)
        np.savetxt(f"{dir_name}/hybrid_sampler-cov_diag_array.txt", cov_diag_array)

accept_rate = num_accepts / N * 100
print("Done.")
print(f"Acceptance rate: {accept_rate:.2f}%")

np.savetxt(f"{dir_name}/hybrid_sampler-paths.txt", samplesHybrid[:, :])
np.savetxt(f"{dir_name}/hybrid_sampler-alpha.txt", samples_alphaHybrid[:])
np.savetxt(f"{dir_name}/hybrid_sampler-sigma.txt", samples_sigmaHybrid[:])
np.savetxt(f"{dir_name}/hybrid_sampler-array_accepts.txt", array_accepts)
# np.savetxt(f"{dir_name}/hybrid_sampler-alpha_variance_array.txt", alpha_variance_array)
np.savetxt(f"{dir_name}/hybrid_sampler-cov_diag_array.txt", cov_diag_array)
with open(f"{dir_name}/hybrid_sampler_info.txt", 'w') as f:
    msg = f"""N = {N}\n\nthin_step={thin_step}\n\nomega={omega}\n\nM={M_trunc}\n\nAcceptance rate: {accept_rate:.1f}%"""
    f.write(msg)

end_time = time.time()
print(f"Running time: {(end_time-start_time)/60:.2f}min")
