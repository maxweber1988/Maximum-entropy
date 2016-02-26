from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset

from max_ent_functions import BCS_spectrum, calc_A, calc_K, root_finding_diag, calc_p_alpha, root_finding_newton

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
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
# s = len(sig_vec[sig_vec>1e-13])

# #reduce all matrices to singular space
# U = U_T.T
# U_s = U[:,0:s]
# V_s = V[:,0:s]
# Sigma_s = Sigma[0:s,0:s]
# K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))

# alpharange = np.array([.5,5.,50.])

# fig,ax = plt.subplots(len(alpharange),1,sharex=True,frameon = False)

# for i in xrange(len(alpharange)):
# 	alpha = alpharange[i]

# 	# create starting values for u and start root finding recursively
# 	u=np.ones(s)
# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)

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
# U = U_T.T
# # create sigma matrix for convenience out of sig_vec
# Sigma = np.diag(sig_vec)

# cutoffs = np.array([1e-2,1e-3,1e-5,1e-10,1e-15])

# fig,ax = plt.subplots(len(cutoffs),1,sharex=True, frameon = False)

# for i in xrange(len(cutoffs)):
# 	# find important singular values and reduce dimensions accordingly
# 	s = len(sig_vec[sig_vec>cutoffs[i]])

# 	#reduce all matrices to singular space
# 	U_s = U[:,0:s]
# 	V_s = V[:,0:s]
# 	Sigma_s = Sigma[0:s,0:s]
# 	K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))
# 	alpha = 1.

# 	# create starting values for u and start root finding recursively
# 	u=np.ones(s)
# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)

# 	A_est = m * np.exp(np.dot(U_s,u_sol))
# 	ax[i].plot(w,A)
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

# # A = calc_A(dw,Nw,[0.],[2.],[1.])
# # calculate K, G
# K = calc_K(dw,Nw,dtau,Ntau,beta)
# G = np.dot(K,A)

# # add relative noise to G
# std = 1e-2 * G
# G_noisy = G + np.random.normal(0.,std,len(G))
# Cov = np.diag(std)

# # singular value decomposition of K = V * Sigma * transpose(U)
# V,sig_vec,U_T = np.linalg.svd(K)

# # create sigma matrix for convenience out of sig_vec
# Sigma = np.diag(sig_vec)

# # find important singular values and reduce dimensions accordingly
# s = len(sig_vec[sig_vec>1e-13])

# #reduce all matrices to singular space
# U = U_T.T
# U_s = U[:,0:s]
# V_s = V[:,0:s]
# Sigma_s = Sigma[0:s,0:s]
# K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))

# alpharange = np.arange(.5,20.1,.1)
# p_alpha = np.zeros(len(alpharange))
# A_mat = np.zeros((Nw,len(alpharange)))
# for i in xrange(len(alpharange)):
# 	print(i)
# 	alpha = alpharange[i]

# 	u=np.ones(s)
# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)

# 	A_est = m * np.exp(np.dot(U_s,u_sol))
# 	A_mat[:,i] = A_est
# 	p_alpha[i] = calc_p_alpha(A_est,alpha,Cov,G_noisy,K_s,m)
# p_alpha = p_alpha/(np.sum(p_alpha))
# # np.save('./data/A_mat.npy',A_mat)
# # np.save('./data/p_alpha.npy',p_alpha)
# p_alpha = np.load('./data/p_alpha.npy')*0.1
# print(np.sum(p_alpha))
# A_mat = np.load('./data/A_mat.npy')
# argmax_alpha = np.argmax(p_alpha)

# A_classic = A_mat[:,argmax_alpha]
# A_Bryan = np.average(A_mat ,axis = 1,weights = p_alpha)
# fig,ax = plt.subplots(2,1,frameon = False)
# ax[0].plot(alpharange,p_alpha)
# ax[0].set_xlabel(r'$\alpha$')
# ax[0].set_ylabel(r'$P_{\alpha}$')
# ax[0].set_xlim(0,20)
# ax[0].axvline(x=alpharange[np.argmax(p_alpha)],ymin=0,ymax = 1.,c='r',label = 'Maximum value at {0}'.format(alpharange[argmax_alpha]))
# ax[0].legend()
# ax[1].plot(w,A_classic,'g',label='classic method')
# ax[1].plot(w,A_Bryan,'r--',label='Bryans method')
# ax[1].plot(w,A,'b',label='original spectrum')
# ax[1].set_xlabel(r'$\omega$')
# ax[1].set_ylabel(r'$A(\omega)$')
# ax[1].set_xlim(0,6)
# ax[1].legend()
# ax[1].set_xlim(0,6)
# plt.tight_layout()
# plt.savefig('../report/images/BCS_Bryan_classic_p_alpha.pdf')
# plt.show()

###########################################################
# varying noise
###########################################################

# W = 10.
# Delta = .9
# alpha = 5.

# # #calculate BCS spectrum
# A = BCS_spectrum(beta,dw,Delta,W)
# # A = calc_A(dw,Nw,[0.],[2.],[1.])
# # calculate K, G
# K = calc_K(dw,Nw,dtau,Ntau,beta)
# G = np.dot(K,A)
# noise_range = np.arange(1.,5.,1.)
# noise = np.random.normal(0,1,len(G))
# fig,ax = plt.subplots(1,1,frameon = False)
# for i in range(len(noise_range)):
# 	#print(noise_range[i])
# 	# add relative noise to G
# 	std = 10 ** (-noise_range[i]) * G
# 	G_noisy = G + noise*std
# 	Cov = np.diag(std)

# 	# singular value decomposition of K = V * Sigma * transpose(U)
# 	V,sig_vec,U_T = np.linalg.svd(K)

# 	# create sigma matrix for convenience out of sig_vec
# 	Sigma = np.diag(sig_vec)
# 	# find important singular values and reduce dimensions accordingly
# 	s = len(sig_vec[sig_vec>1e-10])
# 	print(s)
# 	#reduce all matrices to singular space
# 	U = U_T.T
# 	U_s = U[:,0:s]
# 	V_s = V[:,0:s]
# 	Sigma_s = Sigma[0:s,0:s]
# 	K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))

# 	u=np.ones(s)
# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)

# 	A_est = m * np.exp(np.dot(U_s,u_sol))
# 	ax.plot(w,A_est,label='estimated spectrum noise = 1e-{0}'.format(noise_range[i]))

# 	ax.set_xlabel(r'$\omega$')
# 	ax.set_ylabel(r'$A(\omega)$')
# ax.plot(w,A,label='original spectrum')
# ax.legend()
# ax.set_xlim(0,6)
# plt.tight_layout()

# plt.savefig('../report/images/BCS_varying_noise.pdf')
# plt.show()

############################################################
# influence of m
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
# U = U_T.T
# # create sigma matrix for convenience out of sig_vec
# Sigma = np.diag(sig_vec)
# fig,ax = plt.subplots(2,1,sharex=True,frameon= False)
# fig2,ax2 = plt.subplots(1,1,sharex=True,frameon= False)
# axins0 = zoomed_inset_axes(ax[0], 1.4, loc=7)
# axins1 = zoomed_inset_axes(ax[1], 1.4, loc=7)
# m_old = m
# ax[1].plot(w,A,label=r'$BCS \ spectrum$')
# ax[0].plot(w,A,label=r'$BCS \ spectrum$')
# ax2.plot(w,A,label=r'$BCS \ spectrum$')
# A_old = np.zeros(len(A))
# for i in xrange(4):
# 	print(i)
# 	if i == 0:
# 		ax2.plot(w,m,'--',label=r'$m_{norm}$')
# 	if i == 1:
# 		m = m_old*10
# 		ax2.plot(w,m,'--',label=r'$10 \cdot m_{norm}$')
# 	if i == 2:
# 		m = m_old + np.amax(A)*np.exp(-(w-w[np.argmax(A)])**2/.01**2)
# 		ax2.plot(w,m,'--',label=r'$m_{norm}+\delta(\omega - \omega_{max})$')
# 		fig2.tight_layout()
# 		ax2.legend(loc='best')
# 		fig2.savefig('../report/images/BCS_different_default_models.pdf')
# 	# if i == 3:
# 	# 	m = m_old + np.amax(A)*np.exp(-(w-4)**2/.01**2)
# 	# find important singular values and reduce dimensions accordingly
# 	s = len(sig_vec[sig_vec>1e-13])

# 	#reduce all matrices to singular space
# 	U_s = U[:,0:s]
# 	V_s = V[:,0:s]
# 	Sigma_s = Sigma[0:s,0:s]
# 	K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))
# 	alpha = 4.

# 	# create starting values for u and start root finding recursively
# 	u=np.ones(s)
# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)
# 	A_est = m * np.exp(np.dot(U_s,u_sol))
# 	if i == 0:
# 		ax[0].plot(w,A_est,'r',label=r'$m_{norm}$')
# 		axins0.plot(w,A_est,'r-')
# 	if i == 1:
# 		ax[0].plot(w,A_est,'g--',label=r'$10 \cdot m_{norm}$')
# 		axins0.plot(w,A_est,'g--')
# 	if i == 2:
# 		ax[1].plot(w,A_est,'r',label=r'$m_{norm}+\delta(\omega - \omega_{max})$')
# 		axins1.plot(w,A_est,'r-')
# 		max_value = np.amax(A_est)
# 	# if i == 3:
# 	# 	fig2 = plt.figure(frameon=False)
# 	# 	plt.plot(w,A_est,'r',label=r'$m_{norm}+\delta(\omega - \hat{\omega})$')
# 	# 	plt.xlabel(r'$\omega$')
# 	# 	plt.ylabel(r'$A(\omega)$')
# 	# 	plt.savefig('../report/images/BCS_false_delta_peak_example.pdf')
# axins0.plot(w,A,'b')
# axins1.plot(w,A,'b')
# # specify limits for subplots
# ax[0].set_xlim(0,8)
# w_max = np.argmax(A)
# x1, x2, y1, y2 = 0.8, 1.4, 0.0045, max_value+0.0005  # specify the limits
# axins0.set_xlim(x1, x2) # apply the x-limits
# axins0.set_ylim(y1, y2)

# axins0.set_xticks([])
# axins0.set_yticks([])

# x1, x2, y1, y2 = 0.8, 1.4, 0.0045, max_value+0.0005
# axins1.set_xlim(x1, x2) # apply the x-limits
# axins1.set_ylim(y1, y2)

# axins1.set_xticks([])
# axins1.set_yticks([])
# mark_inset(ax[0], axins0, loc1=2, loc2=3, fc="none", ec="0.5")

# mark_inset(ax[1], axins1, loc1=2, loc2=3, fc="none", ec="0.5")
# ax[0].legend(fontsize=15,loc=10)
# ax[1].legend(fontsize=15,loc=10)
# ax[0].set_ylabel(r'$A(\omega)$')
# ax[0].set_ylim(0,np.amax(A_est)+0.0005)
# ax[1].set_xlabel(r'$\omega$')
# ax[1].set_ylabel(r'$A(\omega)$')
# ax[0].set_xlim(0,8)
# ax[1].set_xlim(0,8)
# #plt.tight_layout()
# fig.savefig('../report/images/BCS_delta_peak_example.pdf')
# plt.show()



###########################################################
# annealing approach
###########################################################

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
# U = U_T.T
# # create sigma matrix for convenience out of sig_vec
# Sigma = np.diag(sig_vec)

# fig,ax = plt.subplots(2,1,frameon=False,sharex=True)
# ax[0].plot(w,A)
# ax[1].plot(w,A)
# step = 1.
# beta = np.arange(.1,5. + step, step)
# beta = beta[::-1]
# for i in xrange(len(beta)):
# 	print(i)
# 	# find important singular values and reduce dimensions accordingly
# 	s = len(sig_vec[sig_vec>1e-10])

# 	#reduce all matrices to singular space
# 	U_s = U[:,0:s]
# 	V_s = V[:,0:s]
# 	Sigma_s = Sigma[0:s,0:s]
# 	K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))
# 	alpha = 4.

# 	# create starting values for u and start root finding recursively
# 	if i == 0:
# 		u=np.ones(s)
# 	else:
# 		u = u_sol + np.random.normal(0,beta[i],len(u_sol))
# 		print(u)

# 	u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)
# 	print(u_sol)

# 	A_est = m * np.exp(np.dot(U_s,u_sol))
# 	ax[0].plot(w,A_est,label='{0}. recursion'.format(i))
# ax[0].legend()
# ax[0].set_ylabel(r'$A(\omega)$')
# ax[0].set_xlim(0,7)
# ax[1].legend()
# ax[1].set_xlabel(r'$\omega$')
# ax[1].set_ylabel(r'$A(\omega)$')
# ax[1].set_xlim(0,7)
# #plt.savefig('../report/images/BCS_annealing_example_new.pdf')
# plt.show()
