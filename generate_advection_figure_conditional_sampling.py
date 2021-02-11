import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np
from copy import deepcopy

from advection_sampler.advection_posterior import logLik, samplePrior, IC_prior_mean, num_pt, x_min, x_max, true_IC


conditional_dict = {}
logpostlist_dict = {}

Nconditional = 50000
uval_list = [0.4,0.5,0.6]
for uval in uval_list:
    standardICsamples = np.zeros((Nconditional, num_pt))
    standarduSamples = np.zeros((Nconditional, 1))
    currentIC = true_IC
    currentU = uval
    currentLogPost = logLik(u=currentU, IC=currentIC)
    standardlogPostList = np.zeros(Nconditional)
    standardlogPostList[0] = currentLogPost
    num_accepts = 0

    omega = 0.04

    for i in range(Nconditional):
        ICProp = np.sqrt(1-omega**2)*(currentIC-IC_prior_mean) + omega*samplePrior() + IC_prior_mean
        uProp = currentU # keep u fixed
        proposed_logPost = logLik(u=uProp, IC=ICProp)
        log_alpha = proposed_logPost - currentLogPost
        if log_alpha > (-np.random.exponential()):
            currentIC = ICProp
            currentU = uProp
            currentLogPost = proposed_logPost
            num_accepts += 1
        else: pass
        standardICsamples[i,:] = currentIC
        standarduSamples[i,:] = currentU
        standardlogPostList[i] = currentLogPost
        if i%5000==0:
            print(f"Iteration {i}/{Nconditional}")

    acceptance_rate = num_accepts / Nconditional * 100
    print(f"Acceptance rate: {acceptance_rate:.1f}%\n")
    conditional_dict[uval] = deepcopy(standardICsamples)
    logpostlist_dict[uval] = deepcopy(standardlogPostList)


rcParams.update({'font.size': 17})
burnin = 100
thin = 200


plt.figure(figsize=(12, 6))

for i in range(burnin, Nconditional, thin):
    plt.plot(np.linspace(x_min,x_max, num_pt), conditional_dict[0.4][i,:], alpha=0.1, c='#CC79A7')
for i in range(burnin, Nconditional, thin):
    plt.plot(np.linspace(x_min,x_max, num_pt), conditional_dict[0.5][i,:], alpha=0.1, c='#56B4E9')
for i in range(burnin, Nconditional, thin):
    plt.plot(np.linspace(x_min,x_max, num_pt), conditional_dict[0.6][i,:], alpha=0.1, c='#009E73')
plt.plot(np.linspace(x_min,x_max, num_pt), true_IC, c='#D55E00', label="true IC", lw=4)

custom_lines = [Line2D([0], [0], color="#CC79A7", lw=4),
                Line2D([0], [0], color="#56B4E9", lw=4),
                Line2D([0], [0], color="#009E73", lw=4),
               Line2D([0], [0], color="#D55E00", lw=4)]
plt.legend(custom_lines, ['$c=0.4$', '$c=0.5$', '$c=0.6$', 'true IC'], loc="upper right")

# plt.title(r"conditional $\rho_0|c$ samples", size=20)
plt.xlabel("x", size=28)
plt.ylabel(r"$\rho_0$", size=24, rotation=90)
# plt.savefig("images/paper_images/advection_conditional_c.png")

plt.show()
