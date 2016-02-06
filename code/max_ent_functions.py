import numpy as np
from scipy.linalg import lu_solve,lu_factor
import matplotlib.pyplot as plt

def calc_A(dw, Nw, mu, sigma,ampl):
	if not isinstance(mu,(list,np.ndarray)) & isinstance(sigma,(list,np.ndarray)) & isinstance(ampl,(list,np.ndarray)):
		raise(TypeError('mu, sigma and ampl should all be either a list or numpy array'))
	if len(mu) != len(sigma) & len(mu) != len(ampl):
		raise(IndexError("mu, sigma and ampl have to have the same length"))
	A = np.zeros((Nw))
	w = np.arange(-Nw*dw/2.,Nw*dw/2.,dw)
	for i in xrange(len(mu)):
		A += ampl[i] * np.exp(-((w - mu[i])**2/sigma[i]**2))
	return A * dw

def calc_K(dw,Nw,dtau,Ntau,beta):

	res = np.zeros((Ntau,Nw))

	for i in xrange(0,Ntau):
		for j in xrange(0,Nw):
			res[i,j] = np.exp(-j * dw * i * dtau) / (1. + np.exp(-beta * (j * dw)))

	return res

def root_finding_newton(u, m, alpha, V, Sigma, U, G, Cov, dw):

	s=len(u)
	max_val = np.sum(m)
	K_s = np.dot(V,np.dot(Sigma,U.T))
	diff = 1.
	count1 = 1
	max_iter = 1000

	while diff > 1e-10 and count1 <= max_iter:
		print count1
		A_appr = dw * m * np.exp(np.dot(U,u))
		inv_cov = (1. / np.diagonal(Cov)**2)
		inv_cov_mat = np.diag(inv_cov)
		dLdF = - inv_cov * (G - np.dot(K_s, A_appr))
		F_u = - alpha * u - np.dot(Sigma,np.dot(V.T,dLdF))
		M = np.dot(Sigma,np.dot(V.T,np.dot(inv_cov_mat,np.dot(V,Sigma))))
		T = np.dot(U.T,np.dot(np.diag(A_appr),U))
		J = alpha * np.diag(np.ones((s))) + np.dot(M,T)
		lu_and_piv = lu_factor(J)
		delta_u = lu_solve(lu_and_piv,F_u)
		count2 = 1
		while np.dot(delta_u.T,np.dot(T,delta_u.T)) > max_val and count2 <= max_iter:
			J = (alpha+count2*1e10) * np.diag(np.ones((s))) + np.dot(M,T)
			if count2 == max_iter:
				print count2
			lu_and_piv = lu_factor(J)
			delta_u = lu_solve(lu_and_piv,F_u)
			count2 +=1
		u_old = u 
		u = u + delta_u
		diff = np.abs(np.sum(u-u_old))
		count1 += 1
	return u	

def root_finding_diag(u, m, alpha, V, Sigma, U, G, Cov, dw):
	s=len(u)
	max_val = np.sum(m)
	T_s = np.dot(V,np.dot(Sigma,U.T))
	diff = 1.
	max_iter1 = 1000
	max_iter2 = 10000
	count1 = 1
	while diff > 1e-10 and count1 < max_iter1:
		# print count1
		f_appr = dw * m * np.exp(np.dot(U,u))
		if np.any(np.isnan(f_appr)):
			return np.zeros(u.shape)
		inv_cov = (1./np.diagonal(Cov))
		inv_cov_mat = np.diag(inv_cov)
		dLdF = - inv_cov * (G - np.dot(T_s, f_appr))
		g = np.dot(Sigma,np.dot(V.T,dLdF))
		F_u = - alpha * u - g
		M = np.dot(Sigma,np.dot(V.T,np.dot(inv_cov_mat,np.dot(V,Sigma))))
		K = np.dot(U.T,np.dot(np.diag(f_appr),U))
		if np.any(np.isnan(K)):
			print "Nan values encountered in K"
			return np.zeros(u.shape)
		eig_K, P = np.linalg.eig(K)
		O = np.diag(eig_K)
		if len(eig_K[eig_K<0.]):
			print eig_K
		A = np.dot(np.sqrt(O), np.dot(P.T, np.dot(M, np.dot(P, np.sqrt(O)))))
		eig_A,R = np.linalg.eig(A)
		Lambda = np.diag(eig_A)
		Y_inv = np.dot(R.T,np.dot(np.sqrt(O),P.T))
		B = (alpha)*np.diag(np.ones((s))) + Lambda
		c_vec = -alpha * np.dot(Y_inv,u)-np.dot(Y_inv,g)
		Y_inv_delta_u = np.zeros(len(c_vec))
		for i in xrange(len(c_vec)):
			Y_inv_delta_u[i] = c_vec[i] / B[i,i]

		delta_u = (-alpha * u - g - np.dot(M,np.dot(Y_inv.T,Y_inv_delta_u)))/(alpha)
		count2 = 1
		while np.dot(delta_u.T,np.dot(K,delta_u)) > max_val and count2 < max_iter2:
			B = (alpha+count2*1.)*np.diag(np.ones((s))) + Lambda
			c_vec = -alpha * np.dot(Y_inv,u)-np.dot(Y_inv,g)
			Y_inv_delta_u = np.zeros(len(c_vec))
			for j in xrange(len(c_vec)):
				Y_inv_delta_u[j] = c_vec[j] / B[j,j]
			delta_u = (-alpha * u - g - np.dot(M,np.dot(Y_inv.T,Y_inv_delta_u)))/(alpha+count2 * 1.)
			count2 += 1
		#print np.dot(delta_u.T,np.dot(K,delta_u))
		u_old = u
		u = u + delta_u
		diff = np.abs(np.sum(u-u_old))
		count1 += 1
	return u
def calc_p_alpha(A,alpha,Cov,G,K,m):
	inv_cov_squ = (np.diag(1./np.diagonal(Cov)))**2
	d2_chi = np.dot(K.T,np.dot(inv_cov_squ,K))
	mat = np.dot(np.diag(np.sqrt(A)),np.dot(d2_chi,np.diag(np.sqrt(A))))
	eig,eigv = np.linalg.eig(mat)
	S = np.sum(A - m - A * np.log(A / m))
	print"S = ", S
	L = 0.5 * np.sum((G - np.dot(K,A))**2/np.diagonal(Cov)**2)
	Q = alpha * S - L
	p_alpha_val = np.prod(np.sqrt(alpha/(alpha+eig)) * 1./alpha * np.exp(Q))
	return p_alpha_val
