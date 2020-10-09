import time
import os
from pathlib import Path
import numpy as np
from advection_sampler.advection_posterior import samplePrior, logLik, IC_prior_mean, num_pt, true_IC, log_prior_u


"""
Sampler for the advection equation:
Joint proposal for the initial condition and the wave speed u: pCN and Gaussian RW
"""



Nstandard = 5000#00000
thin_samples = 5000

if 'global_storage' in os.environ:
    global_storage_path = os.environ['global_storage'] + "/"
else:
    global_storage_path = ""

# Make the relevant folders
dir_name = f"{global_storage_path}outputs/advection_sampler/loss_sd-02-t_1_2/pCN"
Path(f"{global_storage_path}outputs/advection_sampler").mkdir(exist_ok=True)
Path(f"{global_storage_path}outputs/advection_sampler/loss_sd-02-t_1_2").mkdir(exist_ok=True)
Path(dir_name).mkdir(exist_ok=True)


assert Nstandard % thin_samples == 0
N_saved = int(Nstandard/thin_samples)

currentIC = true_IC # samplePrior() + IC_prior_mean
currentU = 0.5
standardICsamples = np.zeros((N_saved, num_pt))
standarduSamples = np.zeros((N_saved, 1))

standardICsamples[0,:] = currentIC
standarduSamples[0] = currentU

currentLogPost = logLik(u=currentU, IC=currentIC) + log_prior_u(currentU)
standardlogPostList = np.zeros(N_saved)
standardlogPostList[0] = currentLogPost
num_accepts = 0

# equally spaced observations, loss_sd=0.5
# omega = 0.04
# u_propSd = 0.0001

# equally spaced observations, loss_sd=0.2
omega = 0.01
u_propSd = 5e-5

# # equally spaced observations, loss_sd=0.1
# omega = 0.005
# u_propSd = 5e-6


# old tuning
# omega = 0.1
# u_propSd = 0.0005

start = time.time()
print(f"Running advection sampler for {Nstandard} iterations with omega={omega} and u_propSd={u_propSd}")
for i in range(1, Nstandard):
    ICProp = np.sqrt(1-omega**2)*(currentIC-IC_prior_mean) + omega*samplePrior() + IC_prior_mean
    uProp = currentU + np.random.normal(0, np.sqrt(u_propSd))
    #uProp = currentU # keep u fixed
    proposed_logPost = logLik(u=uProp, IC=ICProp) + log_prior_u(uProp)
    log_alpha = proposed_logPost - currentLogPost
    if log_alpha > (-np.random.exponential()):
        currentIC = ICProp
        currentU = uProp
        currentLogPost = proposed_logPost
        num_accepts += 1
    else: pass
    if i%thin_samples == 0:
        i_save = int(i/thin_samples)
        standardICsamples[i_save, :] = currentIC
        standarduSamples[i_save, :] = currentU
        standardlogPostList[i_save] = currentLogPost
    if i%5000000==0:
        print(f"Iteration {i}/{Nstandard}")
        print("Saving samples.")
        np.savetxt(f"{dir_name}/standard_IC_samples.txt", standardICsamples)
        np.savetxt(f"{dir_name}/standard_u_samples.txt", standarduSamples)
        print("Done.")

acceptance_rate = num_accepts / Nstandard * 100
print(f"Acceptance rate: {acceptance_rate:.1f}%")
end = time.time()

print(f"Running time: {(end-start)/60:.2f}min")
print("Done sampling.")
# save samples
print("Saving samples.")
np.savetxt(f"{dir_name}/standard_IC_samples.txt", standardICsamples)
np.savetxt(f"{dir_name}/standard_u_samples.txt", standarduSamples)
print("Done.")
