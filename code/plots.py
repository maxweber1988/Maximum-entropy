from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from max_ent_functions import BCS_spectrum, calc_A, calc_K, root_finding_diag, calc_p_alpha

plt.rc('axes',labelsize = 18)
plt.rc('lines',linewidth = 1.5)
beta = 10.
Nw = 500
dw = beta/Nw
Ntau = 1000
dtau = (beta/2) / Ntau
m = np.zeros((Nw)) + 1./Nw

w = np.arange(0.,Nw * dw,dw)
tau = np.arange(0.,Ntau * dtau,dtau)

############################################################
# example plot for A and G in section results
############################################################

# W = 10.
# Delta = .9

# #calculate BCS spectrum
# A = BCS_spectrum(beta,dw,Delta,W)

# # calculate K, G
# K = calc_K(dw,Nw,dtau,Ntau,beta)
# G = np.dot(K,A)

# fig,ax = plt.subplots(1,2,frameon=False)
# ax[0].plot(w,A)
# ax[0].set_xlabel(r'$\omega$')
# ax[0].set_ylabel(r'$A(\omega)$')
# ax[1].plot(tau,G)
# ax[1].set_xlabel(r'$\tau$')
# ax[1].set_ylabel(r'$G(\tau)$')
# plt.tight_layout()
# plt.savefig('../report/images/BCS_A_G_example.pdf')
# plt.show()

############################################################
# illustration of the influence of alpha
############################################################

# W = 10.
# Delta = .9

# #calculate BCS spectrum
# A = BCS_spectrum(beta,dw,Delta,W)

# # calculate K, G
# K = calc_K(dw,Nw,dtau,Ntau,beta)
# G = np.dot(K,A)

# # add relative noise to G
# std = G * 1e-4
# G_noisy = G + np.random.normal(0.,std,len(G))
# Cov = np.diag(std)

# # singular value decomposition of K = V * Sigma * transpose(U)
# V,sig_vec,U_T = np.linalg.svd(K)

# # create sigma matrix for convenience out of sig_vec
# Sigma = np.diag(sig_vec)

# # find important singular values and reduce dimensions accordingly
# s = len(sig_vec[sig_vec>1e-6])

# #reduce all matrices to singular space
# U = U_T.T
# U_s = U[:,0:s]
# V_s = V[:,0:s]
# Sigma_s = Sigma[0:s,0:s]
# K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))

# alpharange = np.array([1.,5.,25.])

# fig,ax = plt.subplots(len(alpharange),1,sharex=True,frameon = False)

# for i in xrange(len(alpharange)):
# 	alpha = alpharange[i]

# 	# create starting values for u and start root finding recursively
# 	u=np.ones(s)
# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw,max_iter1 = 5000)

# 	A_est = m * np.exp(np.dot(U_s,u_sol))
# 	ax[i].plot(w,A,label=r'$True A(\omega)$')
# 	ax[i].plot(w, A_est,label=r'$\alpha = {0}$'.format(alpharange[i]))
# 	ax[i].legend()
# 	ax[i].set_xlim(0,6)
# 	ax[i].set_ylabel(r'$A(\omega)$')
# ax[len(alpharange)-1].set_xlabel(r'$\omega$')
# plt.tight_layout()
# plt.savefig('../report/images/BCS_varying_alpha.pdf')
# plt.show()

############################################################
# illustration of the influence of the singular space size
############################################################

# W = 10.
# Delta = .9

# #calculate BCS spectrum
# A = BCS_spectrum(beta,dw,Delta,W)

# # calculate K, G
# K = calc_K(dw,Nw,dtau,Ntau,beta)
# G = np.dot(K,A)

# # add relative noise to G
# std = G * 1e-4
# G_noisy = G + np.random.normal(0.,std,len(G))
# Cov = np.diag(std)

# # singular value decomposition of K = V * Sigma * transpose(U)
# V,sig_vec,U_T = np.linalg.svd(K)

# # create sigma matrix for convenience out of sig_vec
# Sigma = np.diag(sig_vec)

# cutoffs = np.array([1e-4,1e-8,1e-12])

# fig,ax = plt.subplots(len(cutoffs),1,sharex=True, frameon = False)

# for i in xrange(len(cutoffs)):
# 	# find important singular values and reduce dimensions accordingly
# 	s = len(sig_vec[sig_vec>cutoffs[i]])

# 	#reduce all matrices to singular space
# 	U = U_T.T
# 	U_s = U[:,0:s]
# 	V_s = V[:,0:s]
# 	Sigma_s = Sigma[0:s,0:s]
# 	K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))
# 	alpha = 2.5

# 	# create starting values for u and start root finding recursively
# 	u=np.ones(s)
# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)

# 	A_est = m * np.exp(np.dot(U_s,u_sol))
# 	ax[i].plot(w,A,label=r'$True A(\omega)$')
# 	ax[i].plot(w, A_est,label=r'$\theta = {0}$'.format(cutoffs[i]))
# 	ax[i].legend()
# 	ax[i].set_xlim(0,6)
# 	ax[i].set_ylabel(r'$A(\omega)$')
# ax[len(cutoffs)-1].set_xlabel(r'$\omega$')
# plt.tight_layout()
# plt.savefig('../report/images/BCS_varying_cutoffs.pdf')
# plt.show()
############################################################
# calculate p(alpha)
############################################################

# W = 10.
# Delta = .9

# #calculate BCS spectrum
# A = BCS_spectrum(beta,dw,Delta,W)

# # calculate K, G
# K = calc_K(dw,Nw,dtau,Ntau,beta)
# G = np.dot(K,A)

# # add relative noise to G
# std = G * 1e-4
# G_noisy = G + np.random.normal(0.,std,len(G))
# Cov = np.diag(std)

# # singular value decomposition of K = V * Sigma * transpose(U)
# V,sig_vec,U_T = np.linalg.svd(K)

# # create sigma matrix for convenience out of sig_vec
# Sigma = np.diag(sig_vec)

# # find important singular values and reduce dimensions accordingly
# s = len(sig_vec[sig_vec>1e-6])

# #reduce all matrices to singular space
# U = U_T.T
# U_s = U[:,0:s]
# V_s = V[:,0:s]
# Sigma_s = Sigma[0:s,0:s]
# K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))

# alpharange = np.arange(.05,20.05,.05)
# p_alpha = np.zeros(len(alpharange))
# fig,ax = plt.subplots(2,1,frameon = False)
# A_mat = np.zeros((Nw,len(alpharange)))
# for i in xrange(len(alpharange)):
# 	print(i)
# 	alpha = alpharange[i]

# 	u=np.ones(s)
# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)

# 	A_est = m * np.exp(np.dot(U_s,u_sol))
# 	A_mat[:,i] = A_est
# 	p_alpha[i] = calc_p_alpha(A_est,alpha,Cov,G_noisy,K_s,m)
# p_alpha = p_alpha/np.sum(p_alpha)
# argmax_alpha = np.argmax(p_alpha)
# A_classic = A_mat[:,argmax_alpha]
# A_Bryan = np.average(A_mat,axis = 1,weights = p_alpha)
# ax[0].plot(alpharange,p_alpha)
# ax[0].set_xlabel(r'$\alpha$')
# ax[0].set_ylabel(r'$P_{\alpha}$')
# ax[1].plot(w,A_Bryan,label='Bryans method')
# ax[1].plot(w,A_classic,label='classic method')
# ax[1].plot(w,A,label='original spectrum')
# ax[1].set_xlabel(r'$\omega$')
# ax[1].set_ylabel(r'$A(\omega)$')
# ax[1].legend()
# plt.tight_layout()
# plt.savefig('../report/images/BCS_Bryan_classic_p_alpha.pdf')
# plt.show()