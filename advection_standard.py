import numpy as np
from advection_sampler.advection_posterior import samplePrior, logLik, IC_prior_mean, num_pt


"""
Sampler for the advection equation:
Joint proposal for the initial condition and the wave speed u: pCN and Gaussian RW
"""


Nstandard = 300000

thin_samples = 10
assert Nstandard % thin_samples == 0
N_saved = int(Nstandard/thin_samples)

currentIC = samplePrior() + IC_prior_mean
currentU = 0.5
standardICsamples = np.zeros((N_saved, num_pt))
standarduSamples = np.zeros((N_saved, 1))

standardICsamples[0,:] = currentIC
standarduSamples[0] = currentU

currentLogPost = logLik(u=currentU, IC=currentIC)
standardlogPostList = np.zeros(N_saved)
standardlogPostList[0] = currentLogPost
num_accepts = 0

omega = 0.1
u_propSd = 0.0005

print(f"Running advection sampler for {Nstandard} iterations..")
for i in range(1, Nstandard):
    ICProp = np.sqrt(1-omega**2)*(currentIC-IC_prior_mean) + omega*samplePrior() + IC_prior_mean
    uProp = currentU + np.random.normal(0, np.sqrt(u_propSd))
    #uProp = currentU # keep u fixed
    proposed_logPost = logLik(u=uProp, IC=ICProp)
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
    if i%20000==0:
        print(f"Iteration {i}/{Nstandard}")

acceptance_rate = num_accepts / Nstandard * 100
print(f"Acceptance rate: {acceptance_rate:.1f}%")

print("Done sampling.")
# save samples
print("Saving samples.")
np.savetxt(f"outputs/advection/standard_IC_samples.txt", standardICsamples)
np.savetxt(f"outputs/advection/standard_u_samples.txt", standarduSamples)
