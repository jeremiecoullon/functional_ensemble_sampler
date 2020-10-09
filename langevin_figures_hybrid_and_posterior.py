import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np

from emcee.autocorr import integrated_time
from statsmodels.tsa.stattools import acf

from langevin_functions import solve_Langevin, true_path, le_obs, array_obs_points
from BM_prior import get_KL_weights, x_range

"""
Figures for the Langevin example:

1. Plot showing the slow adaptation of the adaptive hybrid sampler
2. The posterior plots
"""


# ==============================
# 1. Plot showing the slow adaptation of the hybrid sampler
# ==============================
dir_name = "outputs/HEC_cluster/outputs/langevin_sampler_sine4/sigma-4_alpha-12/hybrid_sampler"
cov_diag_array = np.genfromtxt(f"{dir_name}/hybrid_sampler-cov_diag_array.txt")
thin_step_hybrid = 300
N_hybrid = cov_diag_array.shape[0]*thin_step_hybrid
rcParams.update({'font.size': 20})

# look at the first 2 million samples
cut_samples = 2000000

_cut_samples_thin = int(cut_samples/thin_step_hybrid)
fig1, ax = plt.subplots(2, 2, figsize=(14, 6), sharex=True)

parm_list = ['alpha', 'beta'] + [f'coef {e}' for e in range(1,2)]

ax[(0,0)].plot(np.arange(0, N_hybrid, thin_step_hybrid)[:_cut_samples_thin], cov_diag_array[:_cut_samples_thin,0])
ax[(0,0)].set_ylabel('alpha', size=22)

ax[(0,1)].plot(np.arange(0, N_hybrid, thin_step_hybrid)[:_cut_samples_thin], cov_diag_array[:_cut_samples_thin,1])
ax[(0,1)].set_ylabel('beta', size=22)

ax[(1,0)].plot(np.arange(0, N_hybrid, thin_step_hybrid)[:_cut_samples_thin], cov_diag_array[:_cut_samples_thin,2])
ax[(1,0)].set_ylabel('Coef 1', size=22)

ax[(1,1)].plot(np.arange(0, N_hybrid, thin_step_hybrid)[:_cut_samples_thin], cov_diag_array[:_cut_samples_thin,3])
ax[(1,1)].set_ylabel('Coef 2', size=22)

ax[(1,0)].set_xlabel("Iteration", size=20)
ax[(1,1)].set_xlabel("Iteration", size=20)

plt.tight_layout()

# plt.savefig("images/paper_images/langevin_hybrid-variance_adaptation.png")


# ==============================
# Posterior plot
# ==============================
L = 100
dir_name = f"outputs/HEC_cluster/outputs/langevin_sampler_sine4/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/MwG"

thin_step = 200

samplespaths_walker0 = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-paths_L{L}.txt")[:35000]
samplesAlpha = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-alpha_L{L}.txt")[:35000]
samplesSigma = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-sigma_L{L}.txt")[:35000]

N_L = samplespaths_walker0.shape[0]
burnin = 500
plt.rcParams.update({'font.size': 18})

# alpha and sigma posterior pdf
fig2, ax = plt.subplots(2, figsize=(12, 8))
ax[0].hist(np.exp(samplesAlpha[burnin:]), bins=170, color="#56B4E9")
ax[0].set_xlabel("Alpha", size=20)
ax[1].hist(np.exp(samplesSigma[burnin:]), bins=150, color="#56B4E9")
ax[1].set_xlabel("Sigma", size=20)

ax[0].set_xlim((0, 80))
ax[1].set_xlim((0, 10))
plt.tight_layout()
# plt.savefig("images/paper_images/posterior_params.png")
plt.show()

# ========
# X_t paths:
rcParams.update({'font.size': 20})

plot_thin_step = 30

plt.figure(figsize=(10,8))

for i in range(burnin, N_L, plot_thin_step):
    X_t, P_t = solve_Langevin(samplespaths_walker0[i, :], np.exp(samplesAlpha[i]), np.exp(samplesSigma[i]))
    plt.plot(x_range, X_t, alpha=0.01, c='#56B4E9')

plt.scatter(array_obs_points, le_obs, c='#009E73', label="observations", s=80)
plt.plot(x_range, true_path, label="true path", c='#D55E00', lw=4, alpha=0.9)

plt.ylim((-2.7, 2.7))
plt.legend()
plt.xlabel(r"$t$", size=25)
plt.ylabel(r"$X_t$", size=25)


# plt.savefig("images/paper_images/posterior_Xt_paths.png")
plt.show()
