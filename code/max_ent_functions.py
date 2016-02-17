from __future__ import print_function
import numpy as np
from scipy.linalg import lu_solve,lu_factor, eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

def calc_A(dw, Nw, mu, sigma,ampl):
	"""
	Calculates synthetic spectrum A(w) as mixture of Gaussians in the range of [-Nw*dw/2.,Nw*dw/2.].
	The Gaussians are not not normalized and calculated by ampl * np.exp(-((w - mu)**2 / sigma**2))!

	:param dw: step-size for w 
	:param Nw: Number of steps for w
	:param mu: numpy array containing mean values of the single Gaussians
	:param sigma: numpy array containing standard deviation values of the single Gaussians
	:param ampl: numpy array containing amplitudes of the single Gaussians
	:return: numpy array
	"""
	if not isinstance(mu,(list,np.ndarray)) & isinstance(sigma,(list,np.ndarray)) & isinstance(ampl,(list,np.ndarray)):
		raise(TypeError('mu, sigma and ampl should all be either a list or numpy array'))
	if len(mu) != len(sigma) & len(mu) != len(ampl):
		raise(IndexError("mu, sigma and ampl have to have the same length"))
	A = np.zeros((Nw))
	w = np.arange(-Nw*dw/2.,Nw*dw/2.,dw)
	for i in range(len(mu)):
		A += ampl[i] * np.exp(-((w - mu[i])**2/sigma[i]**2))
	A = A * dw
	return A/np.sum(A)

def BCS_spectrum(beta,dw,delta,gap):
	Nw = int(beta/dw)
	spectrum = np.zeros((Nw))
	for i in range(Nw):
		w = i * dw
		if delta < w and w < gap/2.:
			spectrum[i] = np.abs(w)/(np.sqrt(w**2 - delta**2) * gap)
		else:
			spectrum[i] = 0.
	return spectrum / np.sum(spectrum)

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

	s=len(u)
	max_val = np.sum(m)
	K_s = np.dot(V,np.dot(Sigma,U.T))
	diff = 1.
	count1 = 1
	max_iter = 1000

	while diff > 1e-10 and count1 <= max_iter:
		A_appr = m * np.exp(np.dot(U,u))
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
			lu_and_piv = lu_factor(J)
			delta_u = lu_solve(lu_and_piv,F_u)
			count2 +=1
		u_old = u 
		u = u + delta_u
		diff = np.abs(np.sum(u-u_old))
		count1 += 1
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


	s=len(u)
	max_val = np.sum(m)
	K_s = np.dot(V,np.dot(Sigma,U.T))
	diff = 1.
	count1 = 1
	max_iter = 1000

	while diff > 1e-10 and count1 <= max_iter:
		A_appr = m * np.exp(np.dot(U,u))
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
			lu_and_piv = lu_factor(J)
			delta_u = lu_solve(lu_and_piv,F_u)
			count2 +=1
		u_old = u 
		u = u + delta_u
		diff = np.abs(np.sum(u-u_old))
		count1 += 1
	return u

def root_finding_diag(u, m, alpha, V, Sigma, U, G, Cov, dw,max_iter1 = 1000, max_iter2 = 1000):
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
	s=len(u)
	max_val = np.sum(m)
	T_s = np.dot(V,np.dot(Sigma,U.T))
	diff = 1.

	count1 = 1
	u_old = u
	type = np.zeros(max_iter1)
	while diff > 1e-10 and count1 < max_iter1:
		f_appr = m * np.exp(np.dot(U,u))
		f_old = f_appr
		inv_cov = (1. / np.diagonal(Cov)**2)
		inv_cov_mat = np.diag(inv_cov)
		dLdF = - inv_cov * (G - np.dot(T_s, f_appr))
		g = np.dot(Sigma,np.dot(V.T,dLdF))
		F_u = - alpha * u - g
		M = np.dot(Sigma,np.dot(V.T,np.dot(inv_cov_mat,np.dot(V,Sigma))))
		K = np.dot(U.T,np.dot(np.diag(f_appr),U))
		eig_K, P = eigh(K)
		eig_K[eig_K<0.] = 0.
		O = np.diag(eig_K)
		A = np.dot(np.sqrt(O), np.dot(P.T, np.dot(M, np.dot(P, np.sqrt(O)))))
		eig_A,R = eigh(A)
		Lambda = np.diag(eig_A)

		Y_inv = np.dot(R.T,np.dot(np.sqrt(O),P.T))

		B = (alpha)*np.diag(np.ones((s))) + Lambda
		c_vec = -alpha * np.dot(Y_inv,u)-np.dot(Y_inv,g)
		Y_inv_delta_u = np.zeros(len(c_vec))

		for i in range(len(c_vec)):
			Y_inv_delta_u[i] = c_vec[i] / B[i,i]

		delta_u = (-alpha * u - g - np.dot(M,np.dot(Y_inv.T,Y_inv_delta_u)))/(alpha)
		f_appr = m * np.exp(np.dot(U,u+delta_u))
		count2 = 1
		Jac = np.dot(M,K) + np.eye(s) * alpha
		h = 0.1 * np.abs(np.dot(F_u.T,F_u))/np.abs(np.dot(F_u.T,np.dot(Jac,F_u)))
		while np.linalg.norm(f_appr - f_old) > max_val and count2 < max_iter2:
			B = (alpha + count2 * 1./h)*np.diag(np.ones((s))) + Lambda
			c_vec = -alpha * np.dot(Y_inv,u)-np.dot(Y_inv,g)
			Y_inv_delta_u = np.zeros(len(c_vec))
			for j in range(len(c_vec)):
				Y_inv_delta_u[j] = c_vec[j] / B[j,j]
			delta_u = (-alpha * u - g - np.dot(M,np.dot(Y_inv.T,Y_inv_delta_u)))/(alpha + count2 * 1./h)
			f_appr = m * np.exp(np.dot(U,u+delta_u))
			count2 += 1
		u_old = u
		u = u + delta_u
		diff = np.abs(np.sum(u-u_old))
		count1 += 1
	print(count1)
	return u	
	
def calc_p_alpha(A,alpha,Cov,G,K,m):

	inv_cov_squ = (1./np.diagonal(Cov))**2
	# calculate 0.5 * d^2/dA^2 chi


	d2_chi = np.dot(K.T,np.dot(np.diag(inv_cov_squ),K))
	mat = 0.5 * np.dot(np.diag(np.sqrt(A)),np.dot(d2_chi,np.diag(np.sqrt(A))))
	# print(np.amin(mat))
	eig,eigv = eigh(mat)
	# print(eig)
	S = np.sum(A - m - A * np.log((1e-20 + A) / m))
	L = 0.5 * np.sum((G - np.dot(K,A))**2 * inv_cov_squ)
	# print("L=",L,"S=",S)
	Q = alpha * S - L
	#print("e^Q = ",np.exp(Q))
	p_alpha_val = np.prod(np.sqrt(alpha/(alpha+eig))) * 1./alpha * np.exp(Q)
	return p_alpha_val
