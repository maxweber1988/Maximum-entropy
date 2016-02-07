import numpy as np
import matplotlib.pyplot as plt
import time

from max_ent_functions import calc_A, calc_G, calc_K, root_finding_newton, root_finding_diag, max_likelihood_estimate

plt.rc('image',interpolation='nearest',origin='lower')

# define parameters
beta = 10.
Ntau = 200
Nw = 200
dtau = (beta/2.)/Ntau
dw = 10./Nw
m_value = 1./(Nw*dw)
m = np.zeros(Nw) + m_value
# create dummy A
mu = np.array([-1.,1.])
sigma = [1.,2.]
ampl = [2.,1.]
A = calc_A(dw,Nw,mu,sigma,ampl)

plt.plot(A)
# calculate K and G
K = calc_K(dw,Nw,dtau,Ntau,beta)
plt.figure()
plt.imshow(K)
G = np.dot(K,A)
plt.figure()
plt.plot(G)
plt.show()

# transform K,G if necessary and add additional noise to G
G_noisy = G + np.random.normal(0.,1e-5,len(G))

Cov = np.diag(np.ones(len(G))*1e-5)

# singular value decomposition of K = V * Sigma * transpose(U)
V,sig_vec,U_T = np.linalg.svd(K)
# create sigma matrix for convenience out of sig_vec
Sigma = np.diag(sig_vec)

# find important singular values and reduce dimensions accordingly
s = len(sig_vec[sig_vec>1e-6])

U = U_T.T
U_s = U[:,0:s]
V_s = V[:,0:s]
Sigma_s = Sigma[0:s,0:s]
K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))

alpha = .05
# create starting values for u and start root finding recursively
u=np.zeros(s)+ 1.
for j in xrange(100):
	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)
	u = u_sol
# estimated A given solution of root finding
A_est = m * np.exp(np.dot(U_s, u_sol))*dw
plt.plot(A,'-x',label='true A')
plt.plot(A_est,label='estimated A')
plt.legend(loc='best')

A_ML = max_likelihood_estimate(G_noisy,V_s,U_s,Sigma_s)

plt.plot(A_est,'rx',label='A_est')
plt.plot(A,'b+',label='true A')
plt.plot(A_ML,'kx',label='ML A')
plt.legend()


plt.show()
