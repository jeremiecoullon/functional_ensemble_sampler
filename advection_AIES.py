
import numpy as np

from advection_sampler.advection_posterior import samplePrior, IC_prior_mean, num_pt, log_prior_u
from advection_sampler.advection_posterior import cov_prior, precision_mat, data_array, x_min, x_max, loss_sd
from advection_sampler.advectionequation import log_lik_advection
from ou_process import OUProcess



def logPriorIC(IC):
    return - 0.5*np.linalg.multi_dot([IC.T, precision_mat, IC])

def logPost(u, IC):
    """
    log-posterior for both u and IC
    Input centered ICs
    """
    if not (0<u<1.4):
        return -9999999999
    log_lik = log_lik_advection(u=u, IC=IC+IC_prior_mean, data_array=data_array,
                                x_min=x_min, x_max=x_max, error_model='gaussian', loss_sd=loss_sd)
    return log_lik + log_prior_u(u) + logPriorIC(IC)

def invG(u, a):
    phi = 1/(2*(np.sqrt(a) - 1/np.sqrt(a)))
    return (u/(2*phi) + 1/np.sqrt(a))**2

def sampleG(a):
    return invG(np.random.uniform(0,1), a)

# KL transform:
_, evects = np.linalg.eigh(cov_prior)

def get_KL_weights(x):
    return evects.T @ x

def inverseKL(w):
    return evects @ w


# Sampler

L = 17 # number of walkers
N = 20000 # number of iterations
omega = 1 # pCN step size for the complementary subspace

# ============

thin_samples = 10
assert N % thin_samples == 0
N_saved = int(N/thin_samples)

# initialise MCMC: N MCMC iterations, L walkers, num_pt: resolution of function
currentX = np.array([samplePrior() for e in range(L)])
currentU = np.array([np.random.uniform(0,1.4) for e in range(L)])

samples = np.zeros((N_saved, L, num_pt))
samplesU = np.zeros((N_saved, L))


currentLogPost = np.zeros(L)
logPostList = np.zeros((N_saved, L))
for k in range(L):
    currentLogPost[k] = logPost(u=currentU[k], IC=currentX[k,:])
    logPostList[0,k] = currentLogPost[k]
    samples[0,k,:] = currentX[k]
    samplesU[0,k] = currentU[k]


a_prop = 2

def run_MCMC(M_trunc):
    if L < M_trunc+2:
        raise ValueError(f"Number of walkers must be >= {M_trunc+1} (d+2)")

    num_acceptsPCN = 0
    num_acceptsAIES = 0

    # Matrices to project on finite basis and CS:
    project_M_lowfreq = evects[:, -M_trunc:] @ evects[:, -M_trunc:].T
    project_M_highfreq = np.eye(num_pt) - project_M_lowfreq

    print(f"""Running function space AIES for {N} iterations and {L} walkers.\n M_trunc={M_trunc}, and proposal variance a={a_prop}. . pCN omega={omega}\n""")
    for i in range(N):
        for k in range(L):
            # AIES
            j0 = np.random.choice([e for e in range(L) if e!=k])

            w_j0 = get_KL_weights(currentX[j0,:])[-M_trunc:]
            w_k = get_KL_weights(currentX[k,:])
            w_k_start, w_k_end = w_k[:-M_trunc], w_k[-M_trunc:]
            Z = sampleG(a_prop)

            currentparm = np.concatenate([[currentU[k]], w_k_end])
            otherparm = np.concatenate([[currentU[j0]], w_j0])

            uProp, *wProp_end = otherparm*(1-Z) + Z*currentparm
            Xprop = inverseKL(np.concatenate([w_k_start, wProp_end]))
            logPost_prop = logPost(u=uProp, IC=Xprop)
            log_alpha = (M_trunc+1-1)*np.log(Z) + logPost_prop - currentLogPost[k]
            if log_alpha > (-np.random.exponential()):
                currentX[k,:] = Xprop
                currentU[k] = uProp
                currentLogPost[k] = logPost_prop
                num_acceptsAIES += 1
            else: pass

            # pCN
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
            samples[i_save,:,:] = currentX
            samplesU[i_save,:] = currentU
            logPostList[i_save, :] = currentLogPost[:]
        if i%1000 == 0:
            print(f"Iteration {i}/{N}")

    print("Done sampling.")
    acceptance_ratepCN = num_acceptsPCN / (N*L) * 100
    acceptance_rateAIES = num_acceptsAIES / (N*L) * 100
    accept_info = f"N={N}\nthin_samples={thin_samples}\nomega={omega}\nL={L}\n\nAcceptancte rates:\npCN: {acceptance_ratepCN:.1f}%\nensemble: {acceptance_rateAIES:.1f}%"
    # save samples
    print("Saving samples.")
    np.savetxt(f"outputs/advection/IC_samples_M-{M_trunc}-L-{L}-omega-{omega}.txt", np.mean(samples, axis=1))
    np.savetxt(f"outputs/advection/u_samples_M-{M_trunc}-L-{L}-omega-{omega}.txt", np.mean(samplesU, axis=1))
    with open(f"outputs/advection/ensemble_info_M-{M_trunc}-L-{L}-omega-{omega}.txt", "w") as f:
        f.write(accept_info)


    return samples, samplesU, acceptance_ratepCN, acceptance_rateAIES


samples, samplesU, acceptance_ratepCN, acceptance_rateAIES = run_MCMC(M_trunc=15)


print(f"Acceptance rate for pCN: {acceptance_ratepCN:.1f}%")
print(f"Acceptance rate for AIES: {acceptance_rateAIES:.1f}%")
