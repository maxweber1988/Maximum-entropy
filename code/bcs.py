from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from max_ent_functions import calc_K, max_likelihood_estimate, root_finding_diag, calc_p_alpha, BCS_spectrum
import time


beta = 10.
Nw = 300
dw = 10./Nw
m_value = 1./ Nw
m = np.zeros(Nw) + m_value
Ntau = 1000
dtau = (beta / 2) / Ntau

tau = np.arange(0., Ntau * dtau, dtau)
w = np.arange(0., Nw * dw, dw)

A = BCS_spectrum(beta,dw,.9,10.)

K = calc_K(dw,Nw,dtau,Ntau,beta)

G = np.dot(K,A)

Cov = np.diag(G * 1e-4)
G_noisy = G + np.random.normal(0.,np.diagonal(Cov),len(G))

V,sig_vec,U_T = np.linalg.svd(K)

Sigma = np.diag(sig_vec)

# find important singular values and reduce dimensions accordingly
s = len(sig_vec[sig_vec>1e-5])

U = U_T.T
U_s = U[:,0:s]
V_s = V[:,0:s]
Sigma_s = Sigma[0:s,0:s]
K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))
alpharange = np.array([2.])
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
plt.plot(w,A,label = 'true A')
plt.plot(w,A_est,'bx',label='estimated A')
plt.title("estimated A")
plt.ylabel("A")
plt.xlabel('w')
plt.legend()
#plt.plot(w,m*dw,'g+')
plt.show()