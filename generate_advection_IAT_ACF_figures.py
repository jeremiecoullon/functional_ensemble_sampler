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
Figures for the advection example
"""


# ==============================
# Load samples
# ==============================
print("Downloading vanilla samples..")
# standard advection sampler
extra_dir = "outputs/HEC_cluster/"
# extra_dir = ""
standard_samples = np.genfromtxt(f"{extra_dir}outputs/advection_sampler/loss_sd-02-t_1_2/pCN/standard_IC_samples.txt")
standard_samplesU = np.genfromtxt(f"{extra_dir}outputs/advection_sampler/loss_sd-02-t_1_2/pCN/standard_u_samples.txt")
print("Done")
thin_step_pCN = 5000
N_standard, _ = standard_samples.shape

print(f"N_ensemble: {N_standard}")

L = 100


thin_dict = {0:200, 1:200, 5:100, 10:100, 20:100}

samples_dict = {}
print("Only donwload the average over all walkers.")

for M_trunc in [0, 1, 5,10, 20]:
    print(f"Downloading ensemble samples for M={M_trunc}..")
    samples_dict[M_trunc] = {}
    # dir_name = f"outputs/advection_sampler/loss_sd-02-t_1_2/ensemble_sampler/L_{L}/M_{M_trunc}"
    dir_name = f"outputs/HEC_cluster/outputs/advection_sampler/loss_sd-02-t_1_2/ensemble_sampler/L_{L}/M_{M_trunc}"
    samples_dict[M_trunc]['samples'] = np.genfromtxt(f"{dir_name}/IC_samples-L-{L}.txt")
    samples_dict[M_trunc]['samplesU'] = np.genfromtxt(f"{dir_name}/u_samples-L-{L}.txt")

N_ensemble, _ = samples_dict[M_trunc]['samples'].shape
print("Done")
print(f"N_ensemble: {N_ensemble}")

# burnin: 10% of the run length
burnin_dict = {k: int(0.1*v['samplesU'].shape[0]) for k,v in samples_dict.items()}
burnin_pCN = int(0.1*standard_samples.shape[0])


# ==============================
# IAT values
# ==============================
le_tol = 100


print("=====\n")
print(f"IAT for vanilla sampler:")
print(f"c: {thin_step_pCN*integrated_time(standard_samplesU[burnin_pCN:], tol=le_tol)[0]:.0f}")


for basis_number in [1, 5, 15, 100]:
    array_weights = np.array([get_KL_weights(standard_samples[i, :])[-basis_number] for i in range(burnin_pCN, N_standard)])
    print(f"basis element {basis_number}: {thin_step_pCN*integrated_time(array_weights, tol=le_tol)[0]:.0f}")

le_tol = 100


for M_trunc in [0, 1,5,10, 20]:
    print("\n=====\n")
    print(f"IAT for AIES sampler, M_trunc={M_trunc}:")
    N_ensemble = samples_dict[M_trunc]['samples'].shape[0]

    if M_trunc == 0:
        cut_end = 95000 # the sampler for M=0 stopped a bit early, so the last few iterations are all zeros.
    else:
        cut_end = N_ensemble

    print(f"N_ensemble = {N_ensemble*thin_dict[M_trunc]}")
    print(f"c: {thin_dict[M_trunc]*integrated_time(samples_dict[M_trunc]['samplesU'][burnin_dict[M_trunc]:cut_end], tol=le_tol)[0]:.0f}")
    for basis_number in [1, 5, 15, 100]:
        array_weights = np.array([get_KL_weights(samples_dict[M_trunc]['samples'][i, :])[-basis_number] for i in range(burnin_dict[M_trunc], cut_end)])

        print(f"basis element {basis_number}: {thin_dict[M_trunc]*integrated_time(array_weights, tol=le_tol)[0]:.0f}")


# ==============================
# ACF plots
# ==============================
plt.rcParams.update({'font.size': 17})
num_lags = 200

markerlist = ["X", "o", "v", "s", "D", "*"]
markersize = 7
colorlist = ["#CC79A7", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]


x_range = np.arange(0, (num_lags+1)*100, 100)

fig, ax = plt.subplots(2, 1, figsize=(9, 12))

ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
ax[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)

# ============
# ============
# pCN
plot_pCN = ax[0].plot(x_range[::50], acf(standard_samplesU[burnin_pCN:], nlags=num_lags/50, fft=False),
                      label=f"pCN", c=colorlist[0], marker=markerlist[0], markersize=markersize)[0]

basis_number = 1
array_weights = np.array([get_KL_weights(standard_samples[i, :])[-basis_number] for i in range(burnin_pCN, N_standard)])
ax[1].plot(x_range[::50], acf(array_weights, nlags=num_lags/50, fft=False),
             label=f"pCN", c=colorlist[0], marker=markerlist[0], markersize=markersize)


# ============
# ============
# FES
M_trunc = 0
M_samples = samples_dict[M_trunc]['samplesU'][burnin_dict[M_trunc]:95000]
plot_FES_0 = ax[0].plot(x_range[::40], acf(M_samples, nlags=num_lags/2, fft=False)[::20],
                        label=f"M={M_trunc}", c=colorlist[1], marker=markerlist[1], markersize=markersize)
basis_number = 1
array_weights = np.array([get_KL_weights(samples_dict[M_trunc]['samples'][i, :])[-basis_number] for i in range(burnin_dict[M_trunc], M_samples.shape[0])])
ax[1].plot(x_range[::40], acf(array_weights, nlags=num_lags/2, fft=False)[::20],
             label=f"M={M_trunc}", c=colorlist[1], marker=markerlist[1], markersize=markersize)

M_trunc = 1
M_samples = samples_dict[M_trunc]['samplesU'][burnin_dict[M_trunc]:]
plot_FES_1 = ax[0].plot(x_range[::40], acf(M_samples, nlags=num_lags/2, fft=False)[::20],
                        label=f"M={M_trunc}", c=colorlist[2], marker=markerlist[2], markersize=markersize)
basis_number = 1
array_weights = np.array([get_KL_weights(samples_dict[M_trunc]['samples'][i, :])[-basis_number] for i in range(burnin_dict[M_trunc], M_samples.shape[0])])
ax[1].plot(x_range[::40], acf(array_weights, nlags=num_lags/2, fft=False)[::20],
             label=f"M={M_trunc}", c=colorlist[2], marker=markerlist[2], markersize=markersize)


# 5,10,20 are thinned by 100
p_dict_FES = {}
for M_idx, M_trunc in enumerate([5,10,20]):
    M_samples = samples_dict[M_trunc]['samplesU'][burnin_dict[M_trunc]:]
    N_ensemble = M_samples.shape[0]
    p_dict_FES[M_trunc] = ax[0].plot(x_range[::10], acf(M_samples, nlags=num_lags, fft=False)[::10],
                                     label=f"M={M_trunc}", c=colorlist[M_idx+3], marker=markerlist[M_idx+3], markersize=markersize)

    basis_number = 1
    array_weights = np.array([get_KL_weights(samples_dict[M_trunc]['samples'][i, :])[-basis_number] for i in range(burnin_dict[M_trunc], N_ensemble)])
    ax[1].plot(x_range[::10], acf(array_weights, nlags=num_lags, fft=False)[::10],
                 label=f"M={M_trunc}", c=colorlist[M_idx+3], marker=markerlist[M_idx+3], markersize=markersize)


ax[0].set_title("ACF for the wavespeed c", size=20)
ax[1].set_title("ACF for KL coef 1", size=20)

ax[0].set_xlabel("Lags", size=20)
ax[1].set_xlabel("Lags", size=20)

ax[0].set_ylabel("ACF", size=20)
ax[1].set_ylabel("ACF", size=20)

# plt.legend()
fig.legend([plot_pCN, plot_FES_0, plot_FES_1] + list(p_dict_FES.values()),
           labels=['pCN', "M=0", "M=1", "M=5", "M=10", "M=20"],
           loc="upper right",   # Position of legend
#            borderaxespad=-0.2,    # Small spacing around legend box
           bbox_to_anchor=(-0.02,-0.04,1,1),
          prop={'size': 17}
           )
plt.tight_layout()
# plt.savefig("images/paper_images/advection_ACF.png")
plt.show()
