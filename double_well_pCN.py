import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")
from pathlib import Path

from BM_prior import samplePrior, dt, num_pt
from forward_models import solve_DW

"""
Vanilla pCN sampler for double well potential
"""
# ========
# Run pCN
N_pCN = 1000000
thin_step = 100
omega = 0.2



# ============
# ============
# true_BM_sample is a Brownain Motion sample
true_BM_sample = np.genfromtxt("data/conditioned_diff_true_path.txt")
true_path = solve_DW(true_BM_sample)
obs_time_points = list(map(int, np.arange(0.5, 10.5, 0.5,)/dt - 1))
sigma_obs = 0.1
le_obs = true_path[obs_time_points] + np.random.normal(loc=0, scale=sigma_obs, size=20)


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

# ==========
# ==========
N_thin = int(N_pCN/thin_step)

samplespCN = np.zeros((N_thin, num_pt))
currentSamplepCN = true_BM_sample
currentLogLik = loglikelihood(currentSamplepCN)
log_lik_list = np.zeros(N_thin)
log_lik_list[0] = currentLogLik
num_accepts = 0

print(f"Running pCN for {N_pCN} iterations")
for i in range(1, N_pCN):
    xProp = currentSamplepCN*np.sqrt(1-omega**2) + omega*samplePrior()
    prop_LogLik = loglikelihood(xProp)
    AR_log_ratio = prop_LogLik - currentLogLik
    if np.log(np.random.uniform(0,1)) < AR_log_ratio:
        num_accepts += 1
        currentSamplepCN = xProp
        currentLogLik = prop_LogLik
    else: pass
    if i%thin_step==0:
        i_thin = int(i/thin_step)
        log_lik_list[i_thin] = currentLogLik
        samplespCN[i_thin] = currentSamplepCN
    if i%20000==0:
        print(f"Iteration {i}/{N_pCN}")

accept_rate = num_accepts / N_pCN * 100
print("Done.")
print(f"Acceptance rate: {accept_rate:.2f}%")




dir_name = f"outputs/double_well_sampler/pCN"
Path(dir_name).mkdir(exist_ok=True)

# save 0th walker and average over walkers
np.savetxt(f"{dir_name}/vanilla_sampler-paths.txt", samplespCN)
