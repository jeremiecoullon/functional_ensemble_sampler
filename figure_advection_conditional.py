import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np
from copy import deepcopy

from advection_sampler.advection_posterior import logLik, samplePrior, IC_prior_mean, num_pt, x_min, x_max, true_IC


conditional_dict = {}
logpostlist_dict = {}

Nconditional = 2000
uval_list = [0.4,0.5,0.6]
for uval in uval_list:
    standardICsamples = np.zeros((Nconditional, num_pt))
    standarduSamples = np.zeros((Nconditional, 1))
    currentIC = samplePrior() + IC_prior_mean
    currentU = uval
    currentLogPost = logLik(u=currentU, IC=currentIC)
    standardlogPostList = np.zeros(Nconditional)
    standardlogPostList[0] = currentLogPost
    num_accepts = 0

    omega = 0.2
    # u_propSd = 0.0005
    u_propSd = 0.00003

    for i in range(Nconditional):
        ICProp = np.sqrt(1-omega**2)*(currentIC-IC_prior_mean) + omega*samplePrior() + IC_prior_mean
    #     uProp = currentU + np.random.normal(0, np.sqrt(u_propSd))
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



burnin = 100
thin = 10

plt.figure(figsize=(12, 6))

for i in range(burnin, Nconditional, thin):
    plt.plot(np.linspace(x_min,x_max, num_pt), conditional_dict[0.4][i,:], alpha=0.1, c='b')
for i in range(burnin, Nconditional, thin):
    plt.plot(np.linspace(x_min,x_max, num_pt), conditional_dict[0.5][i,:], alpha=0.1, c='g')
for i in range(burnin, Nconditional, thin):
    plt.plot(np.linspace(x_min,x_max, num_pt), conditional_dict[0.6][i,:], alpha=0.1, c='cyan')
plt.plot(np.linspace(x_min,x_max, num_pt), true_IC, c='r', label="true IC")
plt.legend()
plt.title(f"conditional IC samples. c = {uval_list}", size=20)

# plt.savefig("images/paper_images/advection_conditional.png")

plt.show()
