from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from max_ent_functions import calc_K, max_likelihood_estimate, root_finding_diag

beta = 10.
data = np.loadtxt('../blind.txt')

Nw = 800
dw = 10./Nw
m_value = 1./(Nw*dw)
m = np.zeros(Nw) + m_value
Ntau = len(data[:,0])
dtau = data[1,0]-data[0,0]

tau = data[:,0]
w = np.arange(0.,Nw*dw,dw)

G_noisy = data[:,1]
Cov = np.diag(data[:,2])

K = calc_K(dw,Nw,dtau,Ntau,beta)

# singular value decomposition of K = V * Sigma * transpose(U)
V,sig_vec,U_T = np.linalg.svd(K)
# create sigma matrix for convenience out of sig_vec
Sigma = np.diag(sig_vec)

# find important singular values and reduce dimensions accordingly
s = len(sig_vec[sig_vec>1e-5])
print(s)

U = U_T.T
U_s = U[:,0:s]
V_s = V[:,0:s]
Sigma_s = Sigma[0:s,0:s]
K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))

alpha = 1e2
# create starting values for u and start root finding recursively
u=np.ones(s)
u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)

A_est = m * dw * np.exp(np.dot(U_s,u_sol))

plt.figure()
plt.plot(tau,G_noisy)
plt.title("G(tau)")
plt.ylabel("G")
plt.xlabel("tau")
plt.figure()
plt.plot(w,A_est,'bx')
plt.title("estimated A")
plt.ylabel("A")
plt.xlabel('w')
plt.show()
 