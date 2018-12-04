### Log computation of the forward and backward probabilities 
import numpy as np

#%% define the extended helper functions 

def eexp(x):
    """ Extended exponential"""
    if np.isnan(x):
        out = 0
    else:
        out = np.exp(x)
    return out 

def eln(x):
    """ Extended natural log"""
    if x == 0:
        out = np.nan
    elif x > 0:
        out = np.log(x)
    else:
        print("negative input in eln")
    return out

def elnsum(eln_x,eln_y):
    """Extended logarithm sum function"""
    if np.isnan(eln_x) or np.isnan(eln_y):
        if np.isnan(eln_x):
            out = eln_y
        else:
            out = eln_x
    else:
        if eln_x > eln_y:
            out = eln_x + eln(1+np.exp(eln_y-eln_x))
        else:
            out = eln_y + eln(1+np.exp(eln_x-eln_y))
    return out

def elnproduct(eln_x,eln_y):
    """Extended logarithm product"""
    if np.isnan(eln_x) or np.isnan(eln_y):
        out = np.nan
    else:
        out = eln_x + eln_y
    return out 

#%% Computing the probability of observing a sequence

# forward algorithm in log space
def forward_step(A, B, pi, H, K):
    """ Forward step in the log domain"""
    # A is H x H transition matrix
    # B should be N x H matrix of the log pdf's, N is observations
    # B is output of the _log_multivariate_normal_density_diag() function
    # pi is 1 x H vector of prior probabilities
    alpha = np.zeros((H,K))
    for i in range(0,H): # loop through all states at time t = 1
        alpha[i,0] = elnproduct(eln(pi[i]), eln(B[0,i]))
    for t in range(1,K):
        for j in range(H):
            logalpha = np.nan
            for i in range(H):
                tmp = elnproduct(alpha[i,t-1], eln(A[i,j]))
                logalpha = elnsum(logalpha, tmp)
            alpha[j,t] = elnproduct(logalpha, eln(B[t,j]))
    return alpha

def backward_step(A, B, pi, H, K):
    """ Backward step in the log domain"""
    beta = np.zeros((H,K))
    for i in range(H):
        beta[i,K] = 0 
    for t in range(K-2,-1,-1):
        for i in range(H):
            logbeta = np.nan
            for j in range(H):
                tmp1 = elnproduct(B[t+1,j],beta[j,t+1])
                tmp2 = elnproduct(eln(A[i,j]),tmp1)
                logbeta = elnsum(logbeta, tmp2)
            beta[i,t] = logbeta
    return beta
                
def calc_gamma(alpha, beta, H, K):
    """ Calculate the gamma probabilities"""
    gamma = np.zeros((H,K))
    for t in range(K):
        normalizer = np.nan
        for i in range(H):
            gamma[i,t] = elnproduct(alpha[i,t],beta[i,t])
            normalizer = elnsum(normalizer,gamma[i,t])
        for i in range(H):
            gamma[i,t] = elnproduct(gamma[i,t], -normalizer)
    return gamma

def calc_xi(alpha, beta, A, B, H, K):
    """Compute probability of being in state i at time t, and state j at
    time t+1 in log space"""
    xi = np.zeros((K,H,H))
    for t in range(K-1):
        normalizer = np.nan
        for i in range(H):
            for j in range(H):
                tmp1 = elnproduct(eln(B[t+1,j]),beta[j,t+1])
                tmp2 = elnproduct(eln(A[i,j]),tmp1)
                xi[t,i,j] = elnproduct(alpha[i,t],tmp2)
                normalizer = elnsum(normalizer,xi[t,i,j])
        for i in range(H):
            for j in range(H):
                xi[t,i,j] = elnproduct(xi[t,i,j],-normalizer)
    return xi
    
def update_pi(gamma, H):
    pi = []
    for i in range(H):
        pi.append(eexp(gamma[i,0]))
    return pi        
        
def update_A(gamma, xi, H, K):
    A = np.zeros((H,H))
    for i in range(H):
        for j in range(H):
            numerator = np.nan
            denominator = np.nan
            for t in range(K-1):
                numerator = elnsum(numerator, xi[t,i,j])
                denominator = elnsum(denominator, gamma[i,t])
            A[i,j] = eexp(elnproduct(numerator,-denominator))
    return A

        
        
        
        
        
        