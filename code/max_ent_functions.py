import numpy as np
from scipy.linalg import lu_solve,lu_factor

def calc_A(dw,N,peak_indices,peak_width,heights):
	res = np.zeros(N)
	value_number = np.arange(-peak_width/2,peak_width/2+1,1)
	weight_step = 1. / (len(value_number)/2)
	weights = np.arange(-1.,1.+weight_step,weight_step)
	for i in range(len(peak_indices)):
		res[peak_indices[i]-peak_width/2:peak_indices[i]+peak_width/2+1] = heights[i] - heights[i]*np.abs(np.round(weights,8))
	return res

def calc_K(dw,Nw,dtau,Ntau,beta):
	"""calculates the kernel matrix for given number of time steps Ntau, and omega steps Nw.\
	first index gives the time step\
	second index gives the omega step.\
	beta: period of the Greens function, needed for calculatin of the Kernel.\
	dtau,dw: step sizes for imaginary time tau and frequency omega."""

	res = np.zeros((Ntau,Nw))

	for i in range(0,Ntau):
		for j in range(0,Nw):
			res[i,j] = np.exp(-j * dw * i * dtau) / (1. + np.exp(-beta * (j * dw)))
	return res

def calc_G(K,A):
	"""
	:param K: Kernel Matrix
	:param A: Vector representation of the Lehmann spectral function
	:return: resulting greens function given K and A
	"""
	res = np.dot(K,A)
	return res

def root_finding_newton(u, m, alpha, V, Sigma, U, G, Cov, dw):
	"""
	:param u: initial vector of u
	:param m: vector of default model for the Lehmann spectral function
	:param alpha: scalar value, controls the relative weight of maximizing entropie and minimizing kind. of least squares fit.
	:param V: part of singular SVD of K, K = V*Sigma*U.T
	:param Sigma: part of singular SVD of K, K = V*Sigma*U.T
	:param U: part of singular SVD of K, K = V*Sigma*U.T
	:param G: Vector of all average values of the Greensfunction for each time step
	:param Cov: Vector with variance of the re-binned, gaussian distributed QMC approximations for the different time-steps
	:param dw: omega step size
	:return:
	"""

	# dimension of singular space
	s=len(u)
	# maximum iteration step-size, makes sense according to paper of Bryan
	max_val = np.sum(m) #

	## setup matrices constant during the loop calculation
	# recalculate Kernel matrix from its SVD
	K_s = np.dot(V,np.dot(Sigma,U.T))

	# create the inverse covariance matrix
	inv_cov = (1./np.diagonal(Cov))**2
	inv_cov_mat = np.diag(inv_cov)

	# create constant help-matrix for the calculation of the jacobian, J = alpha * 1 + M*X
	M = np.dot(Sigma,np.dot(V.T,np.dot(inv_cov_mat,np.dot(V,Sigma))))

	for i in range(1000):
		## set up matrices varying during the loop
		A_appr = dw * m * np.exp(np.dot(U,u))
			# we need to multiply by dw since in the definition of u the real function A(w) was used and not the discretized
			# form for as used to approximate the integral as a Riemann sum.

		dLdF = - inv_cov * (G - np.dot(K_s, A_appr))
		F_u = -0 * u - np.dot(Sigma,np.dot(V.T,dLdF)) #reset 0 to alpha
		X = np.dot(U.T,np.dot(np.diag(A_appr),U))

		# initialize jacobian
		J = 0 * np.diag(np.ones((s))) + np.dot(M,X) #reset 0 to alpha

		# solve system of linear equations Jdu = -F_u for du
		# using th LU factorization
		lu_and_piv = lu_factor(J)
		delta_u = lu_solve(lu_and_piv,F_u)

		# check if iteration step size is small enough, otherwise adjust parameters such that step size will be smaller
		for j in range(1000):
			if np.dot(delta_u.T,np.dot(X,delta_u)) > max_val:
				J = (alpha+j) * np.diag(np.ones((s))) + np.dot(M,X) # adjusted according to Brayn paper
				lu_and_piv = lu_factor(J)
				delta_u = lu_solve(lu_and_piv,F_u)
			else:
				break
		u_old = u
		u = u + delta_u
		if np.abs(np.sum(u-u_old))<1e-10: # maybe criterion from Brayn paper, p170
			print('Maximum accuracy was reached!')
			break
	return u	


def max_likelihood_estimate(G,V_singular,U_singular,Sigma_singular):
	"""
	:param G: Vector of all average values of the Greensfunction for each time step
	:param V_singular: part of the SVD of K , K = V*Sigma.T*U.T which contains only the components for non zero singular values.
	:param U_singular: part of the SVD of K , K = V*Sigma.T*U.T which contains only the components for non zero singular values.
	:param Sigma_singular: part of the SVD of K , K = V*Sigma.T*U.T which contains only the components for non zero singular values.
	:return: Maximum likelihood estimate of the Lehmann spectral function A.
	"""

	inv_Sigma_singular = np.linalg.inv(Sigma_singular)

	inv_K_singular = np.dot(U_singular,np.dot(inv_Sigma_singular,V_singular.T))

	return np.dot(inv_K_singular,G)


def root_finding_diag(u, m, alpha, V, Sigma, U, G, Cov, dw):
	s=len(u)
	max_val = np.sum(m)
	T_s = np.dot(V,np.dot(Sigma,U.T))
	for i in range(1000):
		f_appr = dw * m * np.exp(np.dot(U,u))
		inv_cov = (1./np.diagonal(Cov))**2
		inv_cov_mat = np.diag(inv_cov)
		dLdF = - inv_cov * (G - np.dot(T_s, f_appr))
		g = np.dot(Sigma,np.dot(V.T,dLdF))
		F_u = - alpha * u - g
		M = np.dot(Sigma,np.dot(V.T,np.dot(inv_cov_mat,np.dot(V,Sigma))))
		K = np.dot(U.T,np.dot(np.diag(f_appr),U))
		eig_K, P = np.linalg.eig(K)
		P_inv = np.linalg.inv(P)
		O = np.diag(eig_K)
		O_inv = np.linalg.inv(O)
		A = np.dot(np.sqrt(O), np.dot(P.T, np.dot(M, np.dot(P, np.sqrt(O)))))
		eig_A,R = np.linalg.eig(A)
		Lambda = np.diag(eig_A)
		Y = np.dot(P,np.dot(np.sqrt(O_inv),R))
		Y_inv = np.linalg.inv(Y)
		for j in range(1000):
			B = (alpha+j)*np.diag(np.ones((s)))+Lambda
			C = -alpha * np.dot(Y_inv,u)-np.dot(Y_inv,g)
			lu_and_piv = lu_factor(B)
			Y_inv_delta_u = lu_solve(lu_and_piv,C)# np.dot(np.linalg.inv(B),C)#
			delta_u = (-alpha * u - g - np.dot(M,np.dot(Y_inv.T,Y_inv_delta_u)))/(alpha+j)
			if np.linalg.norm(np.dot(Y_inv,delta_u)) < max_val:
				break
		u = u + delta_u
	return u
	