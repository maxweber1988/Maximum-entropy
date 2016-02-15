from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from max_ent_functions import calc_K, max_likelihood_estimate, root_finding_diag, calc_p_alpha
import time

beta = 10.
data = np.loadtxt('../blind.txt')
t3 = time.time()
Nw = 800
dw = 10./Nw
m_value = 1./ Nw
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
alpharange = np.arange(1.,10.1,.1)
p_alpha = np.zeros((len(alpharange)))
A_mat = np.zeros((Nw,len(alpharange)))
for i in range(len(alpharange)):
	print(alpharange[i])
	alpha = alpharange[i]
	# create starting values for u and start root finding recursively
	u=np.ones(s)
	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)

	A_est = m * np.exp(np.dot(U_s,u_sol))
	A_mat[:,i] = A_est
	p_alpha[i] = calc_p_alpha(A_est,alpha,Cov,G_noisy,K_s,m*dw)
p_alpha = p_alpha/np.sum(p_alpha)
A_est = np.average(A_mat,axis=1,weights=p_alpha)
print("p_alpha=",p_alpha)
plt.figure()
plt.plot(alpharange,p_alpha)
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
#plt.plot(w,m*dw,'g+')
plt.show()
 