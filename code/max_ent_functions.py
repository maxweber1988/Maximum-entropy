import numpy as np
from scipy.linalg import lu_solve,lu_factor

def calc_A(dw,N,peak_indices,peak_width,heights):
	res = np.zeros(N)
	value_number = np.arange(-peak_width/2,peak_width/2+1,1)
	weight_step = 1. / (len(value_number)/2)
	weights = np.arange(-1.,1.+weight_step,weight_step)
	for i in xrange(len(peak_indices)):
		res[peak_indices[i]-peak_width/2:peak_indices[i]+peak_width/2+1] = heights[i] - heights[i]*np.abs(np.round(weights,8))
	return res

def calc_K(dw,Nw,dtau,Ntau,beta):
	res = np.zeros((Ntau+1,Nw))
	for i in xrange(0,Ntau+1):
		for j in xrange(0,Nw):
			res[i,j] = np.exp(-j * dw * i * dtau) / (1. + np.exp(-beta * (j * dw)))
	return res

def calc_G(K,A,dw):
	res = np.dot(K,A*dw)
	return res
# def integrate_fct(w,mu,sigma,tau,beta):
# 	return (1.-np.abs(w-mu)/mu)*np.exp(-tau*w)/(1.+np.exp((-beta*w)))#np.exp(-(w-mu)**2/sigma**2)

def root_finding_newton(u, m, alpha, V, Sigma, U, G, Cov, dw):
	s=len(u)
	max_val = np.sum(m)
	K_s = np.dot(V,np.dot(Sigma,U.T))
	for i in xrange(1000):
		A_appr = dw * m * np.exp(np.dot(U,u))
		inv_cov = (1./np.diagonal(Cov))**2
		inv_cov_mat = np.diag(inv_cov)
		dLdF = - inv_cov * (G - np.dot(K_s, A_appr))
		F_u = - alpha * u - np.dot(Sigma,np.dot(V.T,dLdF))
		M = np.dot(Sigma,np.dot(V.T,np.dot(inv_cov_mat,np.dot(V,Sigma))))
		T = np.dot(U.T,np.dot(np.diag(A_appr),U))
		J = alpha * np.diag(np.ones((s))) + np.dot(M,T)

		lu_and_piv = lu_factor(J)
		delta_u = lu_solve(lu_and_piv,F_u)
		for j in xrange(1000):
			if np.dot(delta_u.T,np.dot(T,delta_u.T)) > max_val:
				J = (alpha+j) * np.diag(np.zeros((s))+1) + np.dot(M,T)
				lu_and_piv = lu_factor(J)
				delta_u = lu_solve(lu_and_piv,F_u)
			else:
				break
		u = u + delta_u
	return u	

def root_finding_diag(u, m, alpha, V, Sigma, U, G, Cov, dw):
	s=len(u)
	max_val = np.sum(m)
	T_s = np.dot(V,np.dot(Sigma,U.T))
	for i in xrange(1000):
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
		for j in xrange(1000):
			B = (alpha+j)*np.diag(np.ones((s)))+Lambda
			C = -alpha * np.dot(Y_inv,u)-np.dot(Y_inv,g)
			lu_and_piv = lu_factor(B)
			Y_inv_delta_u = lu_solve(lu_and_piv,C)# np.dot(np.linalg.inv(B),C)#
			delta_u = (-alpha * u - g - np.dot(M,np.dot(Y_inv.T,Y_inv_delta_u)))/(alpha+j)
			if np.linalg.norm(np.dot(Y_inv,delta_u)) < max_val:
				break
		u = u + delta_u
	return u

def find_Lambda(A, u, alpha, V, Sigma, U, G, Cov, dw):
	T_s = np.dot(V,np.dot(Sigma,U.T))
	f_appr = A
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
	return Lambda
	