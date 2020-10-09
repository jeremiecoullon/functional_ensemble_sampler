import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np

from emcee.autocorr import integrated_time
from statsmodels.tsa.stattools import acf

from langevin_functions import solve_Langevin, true_path, le_obs, array_obs_points
from BM_prior import get_KL_weights

"""
Get IAT values and create ACF plots for the Langevin example
"""


# ==============================
# Load data
# ==============================
dir_base = "outputs/HEC_cluster/"

# Hybrid
dir_name = f"{dir_base}outputs/langevin_sampler_sine4/sigma-4_alpha-12/hybrid_sampler"
thin_step_hybrid = 300
samplesHybrid = np.genfromtxt(f"{dir_name}/hybrid_sampler-paths.txt")
samples_alphaHybrid = np.genfromtxt(f"{dir_name}/hybrid_sampler-alpha.txt")
samples_sigmaHybrid = np.genfromtxt(f"{dir_name}/hybrid_sampler-sigma.txt")


# Ensemble, joint, L=8
L = 8
dir_name = f"{dir_base}outputs/langevin_sampler_sine4/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/joint_update"
thin_step_ensemble = 200
joint8_samplespaths_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt")
joint8_samplesAlpha_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-alpha_L{L}.txt")
joint8_samplesSigma_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-sigma_L{L}.txt")


# Ensemble, joint, L=100
# this one stopped at 30K
L = 100
dir_name = f"{dir_base}outputs/langevin_sampler_sine4/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/joint_update"
joint100_samplespaths_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt")[:30000]
joint100_samplesAlpha_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-alpha_L{L}.txt")[:30000]
joint100_samplesSigma_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-sigma_L{L}.txt")[:30000]


# Ensemble, MwG
# this one stopped at 35K
L = 100
dir_name = f"{dir_base}outputs/langevin_sampler_sine4/sigma-4_alpha-12/ensemble_sampler/L_{L}-a_2/MwG"
MwG_samplespaths_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-paths_L{L}.txt")[:35000]
MwG_samplesAlpha_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-alpha_L{L}.txt")[:35000]
MwG_samplesSigma_average = np.genfromtxt(f"{dir_name}/ensemble_sampler-average-sigma_L{L}.txt")[:35000]


# Vanilla pCN
dir_name = f"{dir_base}outputs/langevin_sampler_sine4/sigma-4_alpha-12/pCN_sampler/"
thin_step_pCN = 300
samples_pCN = np.genfromtxt(f"{dir_name}/pCN_sampler-paths.txt")
samples_alpha_pCN = np.genfromtxt(f"{dir_name}/pCN_sampler-alpha.txt")
samples_sigma_pCN = np.genfromtxt(f"{dir_name}/pCN_sampler-sigma.txt")



# ==============================
# IAT values
# ==============================
burnin = 100

# alpha:
IAT_joint_8 = integrated_time(joint8_samplesAlpha_average[burnin:])[0]*thin_step_ensemble
IAT_joint_100 = integrated_time(joint100_samplesAlpha_average[burnin:])[0]*thin_step_ensemble
IAT_MwG = integrated_time(MwG_samplesAlpha_average[burnin:])[0]*thin_step_ensemble
IAT_hybrid = integrated_time(samples_alphaHybrid[burnin:])[0]*thin_step_hybrid
IAT_pCN = integrated_time(samples_alpha_pCN[burnin:])[0]*thin_step_pCN

print(f"IAT for alpha")
print(f"pCN: {IAT_pCN:.0f}")
print(f"Hybrid: {IAT_hybrid:.0f}")
print(f"Ensemble joint, L=8: {IAT_joint_8:.0f}")
print(f"Ensemble joint, L=100: {IAT_joint_100:.0f}")
print(f"Ensemble MwG: {IAT_MwG:.0f}")

# sigma:
IAT_joint_8 = integrated_time(joint8_samplesSigma_average[burnin:])[0]*thin_step_ensemble
IAT_joint_100 = integrated_time(joint100_samplesSigma_average[burnin:])[0]*thin_step_ensemble
IAT_MwG = integrated_time(MwG_samplesSigma_average[burnin:])[0]*thin_step_ensemble
IAT_hybrid = integrated_time(samples_sigmaHybrid[burnin:])[0]*thin_step_hybrid
IAT_pCN = integrated_time(samples_sigma_pCN[burnin:])[0]*thin_step_pCN


print("\n====\n")

print(f"IAT for sigma")
print(f"pCN: {IAT_pCN:.0f}")
print(f"Hybrid: {IAT_hybrid:.0f}")
print(f"Ensemble joint, L=8: {IAT_joint_8:.0f}")
print(f"Ensemble joint, L=100: {IAT_joint_100:.0f}")
print(f"Ensemble MwG: {IAT_MwG:.0f}")


# define IAT for KL coefficients of BM paths
le_tol = 100
thin_step_local = 1
print("=====\n")
print(f"Vanilla pCN sampler:")

print("\nFor OU basis elements")

basislist = [1, 10, 100]
IAT_list_pCN = []

def get_path_IAT(samples, basis_number, burnin, thin_step, le_tol=100):
    array_weights = np.array([get_KL_weights(samples[i])[-basis_number] for i in range(burnin, len(samples))])
    IAT = thin_step*integrated_time(array_weights, tol=le_tol)[0]
    return IAT
basis_list = [1, 10, 100]

print("IAT for pCN sampler:")

for basis_number in basis_list:
    IAT_val = get_path_IAT(samples=samples_pCN, basis_number=basis_number,
                burnin=burnin, thin_step=thin_step_pCN, le_tol=100)
    print(f"basis {basis_number}: {IAT_val:.0f}")


print("\nIAT for hybrid sampler:")

for basis_number in basis_list:
    IAT_val = get_path_IAT(samples=samplesHybrid, basis_number=basis_number,
                burnin=burnin, thin_step=thin_step_hybrid, le_tol=100)
    print(f"basis {basis_number}: {IAT_val:.0f}")



print("\nIAT for Ensemble sampler, joint, L=8:")

for basis_number in basis_list:
    IAT_val = get_path_IAT(samples=joint8_samplespaths_average, basis_number=basis_number,
                burnin=burnin, thin_step=thin_step_ensemble, le_tol=100)
    print(f"basis {basis_number}: {IAT_val:.0f}")

print("\nIAT for Ensemble sampler, joint, L=100:")

for basis_number in basis_list:
    IAT_val = get_path_IAT(samples=joint100_samplespaths_average, basis_number=basis_number,
                burnin=burnin, thin_step=thin_step_ensemble, le_tol=100)
    print(f"basis {basis_number}: {IAT_val:.0f}")


print("\nIAT for Ensemble sampler, MwG, L=100:")

for basis_number in basis_list:
    IAT_val = get_path_IAT(samples=MwG_samplespaths_average, basis_number=basis_number,
                burnin=burnin, thin_step=thin_step_ensemble, le_tol=100)
    print(f"basis {basis_number}: {IAT_val:.0f}")


# ==============================
# ACF plots
# ==============================

plt.rcParams.update({'font.size': 18})
markerlist = ["X", "o", "v", "s", "D", "*"]
markersize = 7


num_lags = 50
x_range = np.arange(0, (num_lags+1)*600, 600)

fig, ax = plt.subplots(1,2, figsize=(12, 5))

for idx in range(2):
    ax[idx].ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)


lesams = samples_alpha_pCN[burnin:]
ax[0].plot(x_range[::2], acf(lesams[::2], nlags=num_lags, fft=True)[::2], label="pcn",
           marker=markerlist[0], markersize=markersize)

lesams = samples_alphaHybrid[burnin:]
ax[0].plot(x_range[::2], acf(lesams[::2], nlags=num_lags, fft=True)[::2], label="hybrid",
           marker=markerlist[1], markersize=markersize)



lesams = joint8_samplesAlpha_average[burnin:]
p1 = ax[0].plot(x_range[::2], acf(lesams[::3], nlags=num_lags, fft=True)[::2], label="joint, L=8",
           marker=markerlist[2], markersize=markersize)[0]

lesams = joint100_samplesAlpha_average[burnin:]
ax[0].plot(x_range[::2], acf(lesams[::3], nlags=num_lags, fft=True)[::2], label="joint, L=100",
           marker=markerlist[3], markersize=markersize)

lesams = MwG_samplesAlpha_average[burnin:]
ax[0].plot(x_range[::2], acf(lesams[::3], nlags=num_lags, fft=True)[::2], label="MwG",
           marker=markerlist[4], markersize=markersize)


# ======

num_lags = 25
x_range = np.arange(0, (num_lags+1)*600, 600)

lesams = samples_sigma_pCN[burnin:]
ax[1].plot(x_range, acf(lesams[::2], nlags=num_lags, fft=True), label="pcn",
           marker=markerlist[0], markersize=markersize)

lesams = samples_sigmaHybrid[burnin:]
ax[1].plot(x_range, acf(lesams[::2], nlags=num_lags, fft=True), label="hybrid",
           marker=markerlist[1], markersize=markersize)

lesams = joint8_samplesSigma_average[burnin:]
ax[1].plot(x_range, acf(lesams[::3], nlags=num_lags, fft=True), label="joint, L=8",
           marker=markerlist[2], markersize=markersize)

lesams = joint100_samplesSigma_average[burnin:]
ax[1].plot(x_range, acf(lesams[::3], nlags=num_lags, fft=True), label="joint, L=100",
           marker=markerlist[3], markersize=markersize)

lesams = MwG_samplesSigma_average[burnin:]
ax[1].plot(x_range, acf(lesams[::3], nlags=num_lags, fft=True), label="MwG",
           marker=markerlist[4], markersize=markersize)


ax[1].set_title("ACF for Sigma", size=20)
ax[0].set_title("ACF for Alpha", size=20)

ax[0].set_xlabel("lags",size=20)
ax[0].set_ylabel("ACF",size=20)

ax[1].set_xlabel("lags",size=20)
ax[1].set_ylabel("ACF",size=20)


# plt.legend()
fig.legend([],
           labels=[ "pCN", "hybrid", 'joint, L=8', "joint, L=100", "MwG"],
           loc="center right",
           bbox_to_anchor=(0,0.15,1,1)
           )

plt.tight_layout()
# plt.savefig("images/paper_images/langevin_ACF.png")

plt.show()
