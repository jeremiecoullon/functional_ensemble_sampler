
import numpy as np

T_total = 10
num_pt = 200
dt = T_total/(num_pt-1)
x_range = np.linspace(0, T_total, num_pt)

chol_BM = np.zeros((num_pt, num_pt))

for i in range(1, num_pt):
    for j in range(0, num_pt):
        if i<=j:
            chol_BM[i,j] = np.sqrt(dt)

def samplePrior():
    "sample BM path"
    return np.dot(chol_BM.T, np.random.normal(size=num_pt))


cov_BM = chol_BM.T @ chol_BM
cov_BM[0, 0] = 1e-15 # to prevent the smallest eigenvalue from being 0
precision_BM = np.linalg.inv(cov_BM)

def logPriorBM(x):
    return -0.5*np.linalg.multi_dot([x, precision_BM, x])

# ================
# Functions for the ensemble sampler
#
# # KL transform:
evals, evects = np.linalg.eigh(cov_BM)

def get_KL_weights(x):
    return evects.T @ x

def inverseKL(w):
    return evects @ w

# For ensemble proposal
def invG(u, a):
    phi = 1/(2*(np.sqrt(a) - 1/np.sqrt(a)))
    return (u/(2*phi) + 1/np.sqrt(a))**2

def sampleG(a):
    "For ensemble proposal"
    return invG(np.random.uniform(0,1), a)
