## Viterbi algorithm for determining likelihood 
from scipy.stats import multivariate_normal
import numpy as np

def viterbi(obs, B_mean, B_var, A, pi, H, K):
    # T is the probability of the most likely path of hidden states that would
    # generate the observations seen. T1 is up until time j, T2 is up until time j-1  
    
    # intialize T1 and T2 
    T1 = np.zeros((H,K))
    T2 = np.zeros((H,K))
    for i in range(H):
        emit_pdf = calculate_Gaussian(obs[0,:], B_mean, B_var, i)
        T1[i,0] = pi[i]*emit_pdf
        
    # Calculating T1 and T2 
    for j in range(1,K): # Loop through observations 
        for p in range(H):
            max_transition = T1[0, j-1]*A[0,j]
            for q in range(1, H):
                trans_prob = T1[q, j-1]*A[q,p]
                if trans_prob > max_transition:	
                    max_transition = trans_prob
                    which_state = q
            pdf = calculate_Gaussian(obs[j,:], B_mean, B_var, p)
            transition_probabilities = max_transition * pdf
            T1[p,j] = transition_probabilities
            T2[p,j] = which_state

	max_overall_probability = max(T1[:, K])

	z = [i for i, a in T1[:, K] if a == max_overall_probability]
	hidden_sequence = np.zeros((1,K))
	hidden_sequence[1,K] =  z 

	for i in range(K,1,-1):
		hidden_sequence[1,i-1] = T2[z, i]
		z = T2[z,i]

    return hidden_sequence, max_overall_probability

def calculate_Gaussian(x, Gauss_mean, covar, h):
    # x is a feature vector
    # x_mean is the mean of the Gaussian distributions
    # covar is the covariance matrix
    # h is the hidden state  
    threshold = 1e-3
    pdf = multivariate_normal.pdf(x, mean = Gauss_mean[h,:], cov = np.diag(covar[h,:]))
    return pdf if pdf > threshold else threshold 