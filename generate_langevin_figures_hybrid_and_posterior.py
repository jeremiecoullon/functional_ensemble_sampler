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
fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
ax[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)

ax[0].plot(np.arange(0, N_hybrid, thin_step_hybrid)[:_cut_samples_thin], cov_diag_array[:_cut_samples_thin,0])
ax[0].set_ylabel(r'$\log \alpha$', size=23, rotation=90)
ax[0].yaxis.set_label_coords(-0.1,0.5)

ax[1].plot(np.arange(0, N_hybrid, thin_step_hybrid)[:_cut_samples_thin], cov_diag_array[:_cut_samples_thin,1])
ax[1].set_ylabel(r'$\log \sigma$', size=23, rotation=90)
ax[1].yaxis.set_label_coords(-0.1,0.5)

ax[1].set_xlabel("Iteration", size=23)

plt.tight_layout()
plt.savefig("images/paper_images/langevin_hybrid-variance_adaptation.png")

# ==============================
# Posterior plot
# ==============================
L = 100
dir_name = f"outputs/HEC_cluster/outputs/langevin_sampler_sine4/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/MwG"

thin_step = 200

samplespaths_walker0 = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-paths_L{L}.txt")
samplesAlpha = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-alpha_L{L}.txt")
samplesSigma = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-sigma_L{L}.txt")
burnin_MwG = int(0.1*samplesAlpha.shape[0])

N_L = samplespaths_walker0.shape[0]
plt.rcParams.update({'font.size': 18})

# alpha and sigma posterior pdf
fig, ax = plt.subplots(2, figsize=(12, 8))
ax[0].hist(np.exp(samplesAlpha[burnin_MwG:]), bins=170, color="#56B4E9", density=True)
ax[0].set_xlabel(r"$\alpha$", size=26)
ax[1].hist(np.exp(samplesSigma[burnin_MwG:]), bins=150, color="#56B4E9", density=True)
ax[1].set_xlabel(r"$\sigma$", size=26)

ax[0].set_xlim((0, 55))
ax[1].set_xlim((0, 8))
plt.tight_layout()
# plt.savefig("images/paper_images/posterior_params.png")
plt.show()

# ========
# X_t paths
# ==============================
rcParams.update({'font.size': 20})

plot_thin_step = 30

plt.figure(figsize=(10,8))

for i in range(burnin_MwG, N_L, plot_thin_step):
    X_t, P_t = solve_Langevin(samplespaths_walker0[i, :], np.exp(samplesAlpha[i]), np.exp(samplesSigma[i]))
    plt.plot(x_range, X_t, alpha=0.01, c='#56B4E9')

plt.scatter(array_obs_points, le_obs, c='#009E73', label="observations", s=80)
plt.plot(x_range, true_path, label="true path", c='#D55E00', lw=4, alpha=0.9)
# plt.plot(x_range, another_path, label="true path", c='blue')

plt.ylim((-2.7, 2.7))
plt.legend()
plt.xlabel(r"$t$", size=26)
plt.ylabel(r"$X_t$", size=26, rotation=90)

# plt.savefig("images/paper_images/posterior_Xt_paths.png")
plt.show()
