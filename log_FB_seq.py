### Log computation of the forward and backward probabilities 
import numpy as np
from hmm import cal_b_matrix

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
    """ Forward step in the log domain."""
    # A is H x H transition matrix
    # B should be K x N matrix of the log pdf's, 
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
        beta[i,K-1] = 0 
    for t in range(K-2,-1,-1):
        for i in range(H):
            logbeta = np.nan
            for j in range(H):
                tmp1 = elnproduct(B[t+1,j],beta[j,t+1]) # changed to eln(B)
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

def update_miu(gamma, x, H, K):
    """ Update the means of the Gaussians using 
    one sequence of the training data.
    Returns the elementwise-log of the mean"""
    num = 0
    den = 0
    miu = np.zeros((H,x.shape[1]))
    for i in range(H):
        for t in range(0,K):
            num += eexp(gamma[i,t])*x[t,:]
            den += eexp(gamma[i,t])
        miu[i,:] = np.divide(num,den)
#        miu[i,:] = elnproduct(np.log(num),-den)
    return miu

def update_var(gamma, x, H, K, miu):
    """ Update the covariance matrix using
    one sequence"""
    num = 0
    den = 0
    var = np.zeros((H, x.shape[1]))
    for i in range(H):
        for t in range(0,K):
            num += eexp(gamma[i,t])*np.outer(x[t,:]-miu[i,:], x[t,:]-miu[i,:])
            den += eexp(gamma[i,t])
        var[i,:] = np.diag(np.divide(num,den))
#        var[i,:] = elnproduct(np.lognum,-den).diag()
        
        # set lower bound on variances 
        for j in range(0,x.shape[1]):
            if var[i,j] < 1e-3:
                var[i,j] = 1e-3
    return var
        
def forward_backward_algorithm(x, A, B_mean, B_var, pi, H, K, d):
    """ Performs a full pass through the Baum-Welch algorithm
    and updates A and pi, miu and var. Need to loop through all the 
    sequences, computing alpha, beta, gamma, and xi for each."""
    # input x should be a combination of all the 7 x 561 segments 
    # d is a list of the features to include
    
    # initialize alpha, beta, gamma, and xi matrices 
    E = len(x) # number of sequences
    alpha_mat = []
    beta_mat  = []
    gamma_mat = []
    xi_mat    = []
    
    for e in range(E):
        x_train = x[e][:,d]
        B = cal_b_matrix(x_train, B_mean, B_var, H, K)
        alpha = forward_step(A, B, pi, H, K)
        beta  = backward_step(A, B, pi, H, K)
        gamma = calc_gamma(alpha, beta, H, K)
        xi    = calc_xi(alpha, beta, A, B, H, K)
        
        alpha_mat.append(alpha)
        beta_mat.append(beta)
        gamma_mat.append(gamma)
        xi_mat.append(xi)

    # update pi
    pi_tmp = np.zeros((H,))
    for e in range(E):
        pi_tmp += np.asarray(update_pi(gamma_mat[e], H))
    pi = pi_tmp/E    
        
    # update A
    A = np.zeros((H,H))
    for i in range(H):
        for j in range(H):
            super_num = 0
            super_den = 0
            for e in range(E):
                numerator = np.nan
                denominator = np.nan
                for t in range(K-1):
                    numerator = elnsum(numerator, xi_mat[e][t,i,j])
                    denominator = elnsum(denominator, gamma_mat[e][i,t])
                super_num += eexp(numerator)
                super_den += eexp(denominator)
            A[i,j] = super_num/super_den
    
    # update mean of B
    miu = np.zeros((H,len(d)))
    for i in range(H):
        super_num = 0
        super_den = 0
        for e in range(E):
            num = 0
            den = 0
            for t in range(0,K):
                num += eexp(gamma_mat[e][i,t])*x[e][t,d]
                den += eexp(gamma_mat[e][i,t])
            super_num += num
            super_den += den
        miu[i,:] = np.divide(super_num,super_den)
    
    # update variance of B
    var = np.zeros((H, len(d)))
    for i in range(H):
        super_num = 0
        super_den = 0
        for e in range(E):
            num = 0
            den = 0
            for t in range(0,K):
                num += eexp(gamma_mat[e][i,t])*np.outer(x[e][t,d]-B_mean[i,:], x[e][t,d]-B_mean[i,:])
                den += eexp(gamma_mat[e][i,t])
            super_num += num
            super_den += den
        var[i,:] = np.diag(np.divide(super_num,super_den))
            
        # set lower bound on variances 
        for j in range(0,len(d)):
            if var[i,j] < 1e-6:
                var[i,j] = 1e-6    
    
    return A, miu, var, pi
