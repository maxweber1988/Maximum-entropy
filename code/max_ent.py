import numpy as np
import matplotlib.pyplot as plt
import time
from max_ent_functions import calc_A, calc_G, calc_K, root_finding_newton, root_finding_diag, max_likelihood_estimate


plt.rc('image',interpolation='nearest',origin='lower')
# dummy change
# define parameters
dtau = 0.125
dw = 0.125
Nw = 80
m_value = 1./(Nw*dw)
m = np.zeros(Nw) + m_value
beta = 10.
Ntau = int((beta/2.)  / dtau)
sigma = 2.
# create dummy A and calculate K,G
x = np.arange(0.,10.,dw)
A = np.exp(-((x-3)**2)/0.8)
A += 0.25*np.exp(-((x-6)**2)/2)
A = A * dw
plt.plot(A)
plt.figure()
K = calc_K(dw,Nw,dtau,Ntau,beta)
plt.imshow(K)
G = calc_G(K,A)
plt.figure()
plt.plot(G)
plt.show()
# transform K,G and add additional noise to G
G_noisy = G + np.random.normal(0.,1e-6,len(G))
Cov = np.diag(np.ones(len(G))*1e-4)

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
p_alpha = []
delta_A = []
for i in range(1):
	alpha =  0
	# create starting values for u and start root finding
	u=np.zeros(s)+ 1.
	delta_u = []
	# find solution u recursively, why?
	for j in range(3):
		u_sol= root_finding_newton(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)
		delta_u.append(np.abs(np.sum(u-u_sol)))
		u=u_sol
	A_est = m * np.exp(np.dot(U_s, u_sol)) * dw

	#S = np.sum(A_est - m - np.log(A_est / m))
	#L = 0.5 * np.sum((G_noisy - np.dot(K_s,A_est))**2/np.diagonal(Cov)**2)
	#Q = alpha * S - L
	#p_alpha_val = np.prod(np.sqrt(alpha/(alpha+Lambda) * 1./alpha * np.exp(Q)))
	#p_alpha.append(p_alpha_val)
	#delta_A.append(np.sum(np.sqrt((A-A_est)**2)))

#calculate ML estimate

A_ML = max_likelihood_estimate(G_noisy,V_s,U_s,Sigma_s)

plt.plot(A_est,'rx',label='A_est')
plt.plot(A,'b+',label='true A')
plt.plot(A_ML,'kx',label='ML A')
plt.legend()
#plt.figure()
#plt.plot(p_alpha)
plt.show()
