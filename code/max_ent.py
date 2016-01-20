import numpy as np
import matplotlib.pyplot as plt
import time
from max_ent_functions import calc_A, calc_G, calc_K, root_finding_newton, root_finding_diag,find_Lambda
plt.rc('image',interpolation='nearest',origin='lower')

# define parameters
dtau = 0.125
dw = 0.125
Nw = 80
m_value = dw * (np.zeros(Nw)/Nw)
m = np.zeros(Nw) + 0.1
beta = 10.
Ntau = int(beta / dtau)
sigma=1.5
print Ntau
# create dum
x = np.arange(0.,10.,dw)
A = np.exp(-(x-(beta-5)/2.)**2/sigma**2)
A += 0.5*np.exp(-(x-(beta+5)/2.)**2/sigma**2)
K = calc_K(dw,Nw,dtau,Ntau,beta)
G = calc_G(K,A,dw)

# transform K,G and add additional noise to G
G_noisy = G + np.random.normal(0.,1e-5,len(G))
Cov = np.diag(np.ones(len(G))*1e-5)

# singular value decomposition of K_diag = V * Sigma * transpose(U)
V,sig_vec,U_T = np.linalg.svd(K)

# create sigma matrix for convenience out of sig_vec
Sigma = np.diag(sig_vec)

# find important singular values and reduce dimensions accordingly
s = len(sig_vec[sig_vec>1e-5])
U = U_T.T
U_s = U[:,0:s]
V_s = V[:,0:s]
Sigma_s = Sigma[0:s,0:s]

K_s = np.dot(V_s,np.dot(Sigma_s,U_s.T))
p_alpha = []
delta_A = []
for i in xrange(1):
	print i
	alpha = i + 1.
	print 'alpha = ',alpha 
	# create starting values for u and start root finding
	u=np.zeros(s)+ 1.
	delta_u = []
	# find solution u recusively
	for j in xrange(5):
		print j
		u_sol= root_finding_diag(u = u,m = m, alpha = alpha, V = V_s, Sigma = Sigma_s,U = U_s, G = G_noisy, Cov = Cov, dw = dw)
		delta_u.append(np.abs(np.sum(u-u_sol)))
		u=u_sol
	A_est = m * np.exp(np.dot(U_s, u_sol))
	S = np.sum(A_est - m - np.log(A_est / m))
	L = 0.5 * np.sum((G_noisy - np.dot(K_s,A_est))**2/np.diagonal(Cov)**2)
	Q = alpha * S - L
	Lambda = find_Lambda(A=A_est,u = u_sol, alpha = alpha,V = V_s, Sigma = Sigma_s, U = U_s, G = G_noisy,Cov = Cov, dw = dw)
	p_alpha_val = np.prod(np.sqrt(alpha/(alpha+Lambda) * 1./alpha * np.exp(Q)))
	p_alpha.append(p_alpha_val)
	delta_A.append(np.sum(np.sqrt((A-A_est)**2)))

plt.plot(A_est,'rx',label='A_est')
plt.plot(A,'b+',label='true A')
plt.legend()
plt.figure()
plt.plot(delta_A)
plt.show()