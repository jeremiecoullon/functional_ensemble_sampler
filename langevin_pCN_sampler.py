
import os
import numpy as np
from pathlib import Path
import time

from forward_models import solve_Langevin
from BM_prior import samplePrior, x_range, dt, num_pt, logPriorBM
from langevin_functions import true_path, le_obs, array_obs_points, log_post, sigma_obs


# Proposal covariance fit to a prerun of 500K. for alpha~Exp(12), sigma~Exp(4)

# sin*4
cov_parm_pCN = 0.05*np.array([[1.55795522, 0.7688323 ],
                       [0.7688323 , 0.56861249]])

# V2: prior
# cov_parm_pCN = 0.05*np.array([[1.17901601, 0.52205976],
#                            [0.52205976, 0.55162661]])
N_pCN = 300#00000
omega = 0.08

thin_step = 300


if 'global_storage' in os.environ:
    global_storage_path = os.environ['global_storage'] + "/"
else:
    global_storage_path = ""
dir_name = f"{global_storage_path}outputs/langevin_sampler_sine4/sigma-4_alpha-12/pCN_sampler"
Path(dir_name).mkdir(exist_ok=True)

N_thin = int(N_pCN/thin_step)

samplespCN = np.zeros((N_thin, num_pt))
samples_alphapCN = np.zeros(N_thin)
samples_sigmapCN = np.zeros(N_thin)

currentSamplepCN = samplePrior()
current_alphapCN = 0
current_sigmapCN = 0

currentLogPost = log_post(W=currentSamplepCN, log_alpha=current_alphapCN, log_sigma=current_sigmapCN)
log_post_list = np.zeros(N_thin)
log_post_list[0] = currentLogPost
num_accepts = 0

start_time = time.time()
print(f"Running pCN for {N_pCN} iterations with omega={omega}")
for i in range(1, N_pCN):
    xProp = currentSamplepCN*np.sqrt(1-omega**2) + omega*samplePrior()
    alphaProp, sigmaProp = [current_alphapCN, current_sigmapCN] + np.random.multivariate_normal(mean=[0,0], cov=cov_parm_pCN)
    prop_LogPost = log_post(xProp, alphaProp, sigmaProp)
    AR_log_ratio = prop_LogPost - currentLogPost
    if np.log(np.random.uniform(0,1)) < AR_log_ratio:
        num_accepts += 1
        currentSamplepCN = xProp
        current_alphapCN = alphaProp
        current_sigmapCN = sigmaProp
        currentLogPost = prop_LogPost
    else: pass
    if i%thin_step==0:
        i_thin = int(i/thin_step)
        log_post_list[i_thin] = currentLogPost
        samplespCN[i_thin] = currentSamplepCN
        samples_alphapCN[i_thin] = current_alphapCN
        samples_sigmapCN[i_thin] = current_sigmapCN
    if i%2000000==0:
        print(f"Iteration {i}/{N_pCN}")
        # np.savetxt(f"{dir_name}/pCN_sampler-paths.txt", samplespCN[:, :])
        # np.savetxt(f"{dir_name}/pCN_sampler-alpha.txt", samples_alphapCN[:])
        # np.savetxt(f"{dir_name}/pCN_sampler-sigma.txt", samples_sigmapCN[:])

accept_rate = num_accepts / N_pCN * 100
print("Done.")
print(f"Acceptance rate: {accept_rate:.2f}%")

# np.savetxt(f"{dir_name}/pCN_sampler-paths.txt", samplespCN[:, :])
# np.savetxt(f"{dir_name}/pCN_sampler-alpha.txt", samples_alphapCN[:])
# np.savetxt(f"{dir_name}/pCN_sampler-sigma.txt", samples_sigmapCN[:])
# with open(f"{dir_name}/pCN_sampler_info.txt", 'w') as f:
#     msg = f"""N = {N_pCN}\n\nthin_step={thin_step}\n\nomega={omega}\n\nAcceptance rate: {accept_rate:.1f}%"""
#     f.write(msg)

end_time = time.time()
print(f"Running time: {(end_time-start_time)/60:.2f}min")
