
import numpy as np
import time
from pathlib import Path
import os

from advection_sampler.advection_posterior import samplePrior, num_pt, evects, IC_prior_mean, true_IC
from advection_sampler.advection_posterior import logPriorIC, logPost, invG, sampleG, get_KL_weights, inverseKL


# Sampler

L = 100 # number of walkers
N = 20000000 # number of iterations
# N = 2000 # number of iterations

# =====================
# # TEST SAMPLER: SAMPLE FROM THE PRIOR
# omega = 1
# N = 100000
# thin_step = 2
# L = 8
# # =====================

thin_samples = 200
assert N % thin_samples == 0
N_saved = int(N/thin_samples)

# initialise MCMC: N MCMC iterations, L walkers, num_pt: resolution of function
# currentX = np.array([samplePrior() for e in range(L)])
currentX = np.array([np.sqrt(1-0.9**2)*(true_IC-IC_prior_mean) + 0.9*samplePrior() for e in range(L)])
currentU = np.array([np.random.uniform(0.4,0.6) for e in range(L)])

# samples = np.zeros((N_saved, L, num_pt))
# samplesU = np.zeros((N_saved, L))

# Save only 2 chains. idx 0 is walker 0. idx 1 is the average over all walkers
samples = np.zeros((N_saved, 2, num_pt))
samplesU = np.zeros((N_saved, 2))


currentLogPost = np.zeros(L)
logPostList = np.zeros((N_saved, L))
for k in range(L):
    currentLogPost[k] = logPost(u=currentU[k], IC=currentX[k,:])
    logPostList[0,k] = currentLogPost[k]
    # samples[0,k,:] = currentX[k]
    # samplesU[0,k] = currentU[k]

# walker 0
samples[0, 0,:] = currentX[0]
samplesU[0, 0] = currentU[0]

# average over all walkers
samples[0, 1,:] = np.mean(currentX[:], axis=0)
samplesU[0, 1] = np.mean(currentU)


a_prop = 2


def run_MCMC(M_trunc, omega):
    if L < M_trunc+2:
        raise ValueError(f"Number of walkers must be >= {M_trunc+2} (d+2)")

    # dir_name_base = f"outputs/advection_sampler/ensemble_sampler/L_{L}/"
    # Path(dir_name_base).mkdir(exist_ok=True)
    # dir_name = f"outputs/advection_sampler/ensemble_sampler/L_{L}/M_{M_trunc}"
    # Path(dir_name).mkdir(exist_ok=True)
    if 'global_storage' in os.environ:
        global_storage_path = os.environ['global_storage'] + "/"
    else:
        global_storage_path = ""
    dir_name_base = f"{global_storage_path}outputs/advection_sampler/loss_sd-02-t_1_2/ensemble_sampler/L_{L}"
    Path(dir_name_base).mkdir(exist_ok=True)
    dir_name = f"{global_storage_path}outputs/advection_sampler/loss_sd-02-t_1_2/ensemble_sampler/L_{L}/M_{M_trunc}"
    Path(dir_name).mkdir(exist_ok=True)

    num_acceptsPCN = 0
    num_acceptsAIES = 0
    num_proposals_AIES = 0
    num_proposals_pCN = 0
    # TODO: Fix acceptance rate calculation (MwG sampler)

    # Matrices to project on finite basis and CS:
    if M_trunc>0:
        project_M_lowfreq = evects[:, -M_trunc:] @ evects[:, -M_trunc:].T
    elif M_trunc==0:
        project_M_lowfreq = np.zeros(num_pt)
    else:
        raise ValueError("'M_trunc' must be positive")
    project_M_highfreq = np.eye(num_pt) - project_M_lowfreq

    print(f"""Running function space AIES for {N} iterations and {L} walkers.\n M_trunc={M_trunc}, and proposal variance a={a_prop}. pCN omega={omega}\n""")
    start = time.time()
    for i in range(N):
        for k in range(L):
            if i%2==0:
                # AIES
                num_proposals_AIES += 1
                j0 = np.random.choice([e for e in range(L) if e!=k])
                if M_trunc>0:
                    w_j0 = get_KL_weights(currentX[j0,:])[-M_trunc:]
                    w_k = get_KL_weights(currentX[k,:])
                    w_k_start, w_k_end = w_k[:-M_trunc], w_k[-M_trunc:]
                    Z = sampleG(a_prop)

                    currentparm = np.concatenate([[currentU[k]], w_k_end])
                    otherparm = np.concatenate([[currentU[j0]], w_j0])

                    uProp, *wProp_end = otherparm*(1-Z) + Z*currentparm
                    Xprop = inverseKL(np.concatenate([w_k_start, wProp_end]))
                    logPost_prop = logPost(u=uProp, IC=Xprop)
                elif M_trunc==0:
                    Z = sampleG(a_prop)
                    uProp = currentU[j0]*(1-Z) + Z*currentU[k]
                    Xprop = currentX[k,:]
                    logPost_prop = logPost(u=uProp, IC=Xprop)
                else:
                    raise ValueError("'M_trunc' must be positive")

                log_alpha = (M_trunc+1-1)*np.log(Z) + logPost_prop - currentLogPost[k]
                if log_alpha > (-np.random.exponential()):
                    currentX[k,:] = Xprop
                    currentU[k] = uProp
                    currentLogPost[k] = logPost_prop
                    num_acceptsAIES += 1
                else: pass
            elif i%2==1:
                # pCN
                num_proposals_pCN += 1
                xProp = (project_M_lowfreq + np.sqrt(1-omega**2)*project_M_highfreq)@currentX[k,:] + omega*project_M_highfreq@samplePrior()
                logPost_prop = logPost(u=currentU[k], IC=xProp)
                log_alpha = logPost_prop - currentLogPost[k] - logPriorIC(xProp) + logPriorIC(currentX[k,:])
                if log_alpha > (-np.random.exponential()):
                    currentX[k,:] = xProp
                    currentLogPost[k] = logPost_prop
                    num_acceptsPCN += 1
                else: pass

        # update chain
        if i%thin_samples == 0:
            i_save = int(i/thin_samples)
            # samples[i_save,:,:] = currentX
            # samplesU[i_save,:] = currentU

            # walker 0
            samples[i_save,0,:] = currentX[0,:]
            samplesU[i_save,0] = currentU[0]
            # average over walkers:
            samples[i_save,1,:] = np.mean(currentX[:,:], axis=0)
            samplesU[i_save,1] = np.mean(currentU)

            logPostList[i_save, :] = currentLogPost[:]
        if i%1000000 == 0:
            print(f"Iteration {i}/{N}")
            np.savetxt(f"{dir_name}/IC_samples-L-{L}.txt", samples[:,1,:])
            np.savetxt(f"{dir_name}/u_samples-L-{L}.txt", samplesU[:,1])

            np.savetxt(f"{dir_name}/IC_samples-L-{L}_walker0.txt", samples[:,0, :])
            np.savetxt(f"{dir_name}/u_samples-L-{L}_walker0.txt", samplesU[:,0])


    print("Done sampling.")
    end = time.time()
    print(f"Running time: {(end-start)/60:.1f}min")
    acceptance_ratepCN = num_acceptsPCN / num_proposals_pCN * 100
    acceptance_rateAIES = num_acceptsAIES / num_proposals_AIES * 100
    accept_info = f"N={N}\nthin_samples={thin_samples}\nomega={omega}\nL={L}\n\nAcceptancte rates:\npCN: {acceptance_ratepCN:.1f}%\nensemble: {acceptance_rateAIES:.1f}%"
    # save samples
    print("Saving samples.")
    np.savetxt(f"{dir_name}/IC_samples-L-{L}.txt", samples[:,1,:])
    np.savetxt(f"{dir_name}/u_samples-L-{L}.txt", samplesU[:,1])

    np.savetxt(f"{dir_name}/IC_samples-L-{L}_walker0.txt", samples[:,0, :])
    np.savetxt(f"{dir_name}/u_samples-L-{L}_walker0.txt", samplesU[:,0])

    with open(f"{dir_name}/ensemble_info-L-{L}.txt", "w") as f:
        f.write(accept_info)

    return samples, samplesU, acceptance_ratepCN, acceptance_rateAIES

# Tuning parameters
# M=20, omega=1
# M=10, omega=0.6
# M=5, omega=0.15
# M=1, omega=0.05
# M=0, omega=0.04

samples, samplesU, acceptance_ratepCN, acceptance_rateAIES = run_MCMC(M_trunc=0, omega=0.04)


print(f"Acceptance rate for pCN: {acceptance_ratepCN:.1f}%")
print(f"Acceptance rate for AIES: {acceptance_rateAIES:.1f}%")
