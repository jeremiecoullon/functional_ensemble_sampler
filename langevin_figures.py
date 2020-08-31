import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np

from emcee.autocorr import integrated_time
from langevin_functions import solve_Langevin, true_path, le_obs, array_obs_points

# Hybrid
dir_name = "outputs/langevin_sampler/sigma-4_alpha-12/hybrid_sampler"
thin_step_hybrid = 200
samplesHybrid = np.genfromtxt(f"{dir_name}/hybrid_sampler-paths.txt")
samples_alphaHybrid = np.genfromtxt(f"{dir_name}/hybrid_sampler-alpha.txt")
samples_sigmaHybrid = np.genfromtxt(f"{dir_name}/hybrid_sampler-sigma.txt")


# Ensemble, joint, L=8
L = 8
dir_name = f"outputs/langevin_sampler/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/joint_update"
thin_step_ensemble = 100
joint8_samplespaths_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt")
joint8_samplesAlpha_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-alpha_L{L}.txt")
joint8_samplesSigma_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-sigma_L{L}.txt")


# Ensemble, joint, L=100
L = 100
dir_name = f"outputs/langevin_sampler/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/joint_update"
joint100_samplespaths_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt")
joint100_samplesAlpha_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-alpha_L{L}.txt")
joint100_samplesSigma_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-sigma_L{L}.txt")


# Ensemble, MwG
L = 100
dir_name = f"outputs/langevin_sampler/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/MwG_equal_prob"
MwG_samplespaths_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt")
MwG_samplesAlpha_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-alpha_L{L}.txt")
MwG_samplesSigma_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-sigma_L{L}.txt")



# Vanilla pCN
dir_name = "outputs/langevin_sampler/sigma-4_alpha-12/pCN_sampler/"
thin_step_pCN = 100
samples_pCN = np.genfromtxt(f"{dir_name}/pCN_sampler-paths.txt")
samples_alpha_pCN = np.genfromtxt(f"{dir_name}/pCN_sampler-alpha.txt")
samples_sigma_pCN = np.genfromtxt(f"{dir_name}/pCN_sampler-sigma.txt")


burnin = 500
#
# # IAT values:
IAT_joint_8 = integrated_time(joint8_samplesAlpha_average[burnin:])[0]*thin_step_ensemble
IAT_joint_100 = integrated_time(joint100_samplesAlpha_average[burnin:])[0]*thin_step_ensemble
IAT_MwG = integrated_time(MwG_samplesAlpha_average[burnin:])[0]*thin_step_ensemble
IAT_hybrid = integrated_time(samples_alphaHybrid[burnin:])[0]*thin_step_hybrid
IAT_pCN = integrated_time(samples_alpha_pCN[burnin:])[0]*thin_step_pCN


# Print IAT values
print(f"pCN: {IAT_pCN:.0f}")
print(f"Ensemble joint, L=8: {IAT_joint_8:.0f}")
print(f"Hybrid: {IAT_hybrid:.0f}")
print(f"Ensemble joint, L=100: {IAT_joint_100:.0f}")
print(f"Ensemble MwG: {IAT_MwG:.0f}")


# Plots for posterior samples

L = 100
dir_name = f"outputs/langevin_sampler/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/MwG_equal_prob"
thin_step = 100

samplespaths_walker0 = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-paths_L{L}.txt")
samplesAlpha = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-alpha_L{L}.txt")
samplesSigma = np.genfromtxt(f"{dir_name}/ensemble_sampler-walker0-sigma_L{L}.txt")



N_L = samplespaths_walker0.shape[0]

from BM_prior import x_range
# rcParams.update({'font.size': 13})
fig, ax = plt.subplots(2, figsize=(12, 8))
ax[0].hist(np.exp(samplesAlpha[burnin:]), bins=170)
ax[0].set_xlabel("Alpha", size=20)
ax[1].hist(np.exp(samplesSigma[burnin:]), bins=150)
ax[1].set_xlabel("Sigma", size=20)

plt.savefig("images/paper_images/posterior_params.png")


rcParams.update({'font.size': 16})

plot_thin_step = 10

plt.figure(figsize=(10,8))

for i in range(burnin, N_L, plot_thin_step):
    X_t, P_t = solve_Langevin(samplespaths_walker0[i, :], np.exp(samplesAlpha[i]), np.exp(samplesSigma[i]))
    plt.plot(x_range, X_t, alpha=0.005, c='g')


plt.scatter(array_obs_points, le_obs, c='r', label="observations", s=50)
plt.plot(x_range, true_path, label="true path", c='blue')
plt.legend()
# plt.xlabel("x", size=20)

plt.savefig("images/paper_images/posterior_paths.png")
plt.show()
