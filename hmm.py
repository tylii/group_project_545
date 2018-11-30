## Implementing Hidden Markov Model using Baum-Welch algorithm
# Load the data 

# "subject_train.txt" shows which subject number did the acts in "X_train"
# ^ the rows of "subject_train" match up with the rows of "X_train"
# ^ "y_test" gives the movement of each row from 1-6. NOTE: movements of 1 person can change during experiment
# ^ 1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS, 4 SITTING, 5 STANDING, 6 LAYING

 
import os
import numpy as np 
from sklearn import preprocessing
from scipy import stats
import initialize_hmm
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.stats import multivariate_normal
import argparse

def main():
  # parse commandline arguments
  '''
  ap = argparse.ArgumentParser()
  ap.add_argument("-f","--file", type=str,default = "standing.txt",
    help="the input file which contains the sequences for one (class) of activity")
  ap.add_argument("-t","--len_time", type=int,default = 7,
    help="the length of the observation sequence")
  ap.add_argument("-k", "--k_states", type=int, default = 3,
    help="the number of hidden states")
    
  args = vars(ap.parse_args())
  '''

  args = thisdict =	{
  "file": "standing.txt",
  "h_states": 3,
  "k_states": 7
}  
  input_file = args["file"]
  K = args["k_states"]  # number of observations in a sequence = states  (default = 7)
  H = args["h_states"]   # number of hidden states (default = 3)
  n_iteration = 5

  # load the data
  x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()

  # standardize it (get z-scores)
  x_train = initialize_hmm.standardize_data(x_train) 
  
  # use the first two features for debugging purpose
  # x_train = x_train[:,0:2]

  # initialize the model parameters
  A, pi = initialize_hmm.init_par(x_train, y_train, H)
  kmeans, B_mean, B_var = initialize_GMM(x_train, H)
  
  # get the indices of each activity sequence 
  activity_train = initialize_hmm.segment_data(y_train)  
  activity_test  = initialize_hmm.segment_data(y_test)
  # activity_train/test has three columns: activity   start    end
  
#  states, valid = activity_sequence(5, activity_train, x_train, K)
  segments = all_sequences(x_train,5, activity_train)

  x_train = segments
  #-----------------------------------------#
	## Baum-Welch algorithm 
  ## Step 1: Initialize all Gaussian distributions with the mean and variance along the whole dataset.
  ## Step 2: Calculate the forward and backward probabilities for all states j and times t.

  
  for i in range(n_iteration):
    alpha,beta = forward_backward(x_train, A, B_mean, B_var, pi.T, H)

    # scale alpha and beta 
    for n in range(len(x_train)):
      alpha[n], beta[n] = scale_prob(alpha[n], beta[n],H,K)
    
    A, B_mean, B_var = update_GMM(x_train,alpha,beta,H,A,B_mean,B_var)    
  

def forward_backward(x, A, B_mean,B_var, pi, K):
  # **** In this function K is number of hidden states and T is length of sequence *****
	# calculate the forward probabilities (alpha)
	# alpha[n][i,t] is the joint probability of seeing x_1, x_2,... x_t (observations) and being in state i at time t, for the n-th sequence
	# size of alpha: K * T
  
  # K: number of hidden states
  # x: a list of 2-D matrix. x[sequence ID][timeT,featureID], the features at each "time point" in each time series.
  #                            these IDs all start from zero
  # y: a list of 1-D vector. y[sequence ID] labels corresponding to one sequence

  # alpha[n][k,t]: n is the index of the sequence, k is the index

   # length of the sequence
  
  alpha = defaultdict() # the forward probability
  beta = defaultdict()  # the backward probability

  for n in range(len(x)): # iterate over each sequence to calculate alpha for each sequence
    
    T = np.shape(x[n])[0]
    alpha[n] = np.zeros([K,T])
    beta[n] = np.zeros([K,T])
    for k in range(K):
      # ----- initlaize the first alpha alpha[n][k,0] ------- 
      b_k_o1 = cal_b(x[n][0,:],B_mean[k,:],B_var[k,:])   # fill this in: b[k][x[0]] the probability
      alpha[n][k,0] = pi[k]*b_k_o1
      # ----- initlaize the last beta beta[n][k,T-1] ------- 
      beta[n][k,T-1] = 1

    # calcualte the other alpha values in an iterative way
    for t in range(1,T): # loop over all other time points
      for k in range(K):
        b_k_ot = cal_b(x[n][t,:],B_mean[k,:],B_var[k,:])
        alpha[n][k,t] = alpha[n][:,t-1].T.dot(A[:,k])*b_k_ot

    # calcualte the other beta values in an iterative way
    for t in reversed(range(0,T-1)): # loop over all other time points
      for k in range(K):
        tmp = 0
        for j in range(K):
          b_j_ot = cal_b(x[n][t,:],B_mean[k,:],B_var[k,:])
          tmp += A[k,j]*b_j_ot
        beta[n][k,t] = tmp*beta[n][k,t+1]

  return alpha, beta

## Step 3: For each state j and time t, use the probability Lj(t) and the current observation vector Ot to update the accumulators for that state.

## Step 4: Use the final accumulator values to calculate new parameter values.

## Step 5: If the value of P = P(O/M) for this iteration is not higher than the value at the previous iteration then stop, repeat
## the above steps using the new re-estimated parameter
## values (from step 2) 

#-----------------------------------------#

def scale_prob(alpha, beta,K,T):   
  for k in range(0, K):
    c  = 1/np.sum(alpha[k,:])
    c2 = 1/np.sum(alpha[k,:])
        
    for i in range(0,T):
      alpha[k,i] = alpha[k,i] * c
      beta[k,i] = beta[k,i] * c2
  return alpha, beta

def calc_emission_initial(x, K):
  mean, cov_matrix = compute_B_initial(K)
  pdf = stats.multivariate_normal.pdf(x, mean, cov_matrix)
  return pdf

def compute_B_initial(k):
  indices  = [i for i, a in enumerate(y_train) if a == k]
  x  = [x_train[j] for j in indices]  
  x = np.asarray(x)
  avg_x  = np.mean(x, axis = 0)
  var_x  = np.cov(x.T)
  return avg_x[0:14], var_x[0:14,0:14] # figure this out 

def activity_sequence(i, activityIndex, x, K):
  # i is the index of the activity sequence (400 instances in total)
  # activityIndex is the matrix generated by segment function 
  # x is the associated data set (training or testing)
  # K is the number of states per activity sequence (7 in the literature)
  
  # Returns a (7,561) array, where each row is the average of a certain number 
  # of frames. The number of frames averaged is in the variable frames_per_state.
  
  # get the specific frames, and the length of the activity
  start_ind  = activityIndex[i, 1]
  end_ind    = activityIndex[i, 2]
  length     = end_ind - start_ind
  frame_data = x[start_ind:end_ind, :]
  
  # need to determine how many frames are in each state
  if length >= K:
    frames_per_state = length//K
#    print('~{} frames per state.'.format(frames_per_state))
    # segment the frame data (average the vectors?)
    states = []
    index = 0
    for j in range(0, K):
      data = np.mean(frame_data[index:index + frames_per_state,:], axis = 0)
      states.append(data)
      index += frames_per_state
    states = np.asarray(states)
    valid = True
  else:
    print("Activity is not long enough.")
    states = []
    valid = False
  return states, valid


def cal_b(x,miu,covar):
  # This function calculates the emission probability given the 
  # means and covariance matrix for a multivariate Gaussian

  # miu is a vector of means for one multivariate gaussian 
  # cov is the covariance matrix for one multivariate gaussian
  # x is one obsevation 

  # check if we are using only the diagnal elements of the cov matrix

  if covar.ndim==1:
    return np.log(multivariate_normal.pdf(x, mean=miu, cov=np.diag(covar)))

  return np.log(multivariate_normal.pdf(x, mean=miu, cov=covar))




def all_sequences(x_train, L, segments):
    # Run activity_sequence on all the datas of label 
    # segments is the output of initialize_hmm.segment_data()
    # Returns a list of (7,561) dimensional arrays 
    data_sequence = []
    for i in range(len(segments)):
        if segments[i,0] == L:
            states, valid = activity_sequence(i, segments, x_train, 7)
            if valid: 
                data_sequence.append(states)
    return data_sequence

def initialize_GMM(x, n_Gauss):
  # Use k-means on the input data to determine the initial centers and variances of the Gaussians
  # perform k-means on the entire data set, with number of clusters equal to number of hidden states 
  main_kmeans = KMeans(n_clusters = n_Gauss, random_state = 1).fit(x)  
  Gauss_means = main_kmeans.cluster_centers_
  
  n_feature = x.shape[1] # 561 if we are using all features
  # compute covariance using clustered samples (diagonal matrix, n_components by n_features )
  labels = main_kmeans.labels_
  covar  = np.zeros((n_Gauss, n_feature))
  for i in range(n_Gauss):
    x_clus = x[np.where(labels==i)]
    x_covar  = np.var(x_clus.T, axis = 1)
    
    # set lower bound in the Gaussians
    for j in range(n_feature):
        if x_covar[j] < 1e-3:
            x_covar[j] = 1e-3
            
    covar[i,:] = x_covar
  return main_kmeans, Gauss_means, covar

def update_GMM(x,alpha,beta,H,A, B_mean,B_var):
  K = H
  # --- calcualte gamma ----
  # which is the probability of being in hidden state k at time t for a given sequence
  N = len(x) # number of training sequences
  F = x[0].shape[1] # number of features
  T = x[0].shape[0] # number of time points
  gamma = np.zeros([N,K,T])
  for n in range(N):
    T = x[n].shape[0] # the length of the observation sequence/ time series
    for t in range(T):
      tmp = alpha[n][:,t].T.dot(beta[n][:,t])
      for k in range(K):
        gamma[n,k,t] = alpha[n][k,t]*beta[n][k,t]/tmp

  # ---  Update B (mean and covariance matrix) ----
  # --- update mean ---
  # recalculate mean and variance with weighted average 

  new_mean = np.zeros([K,F])
  new_var = np.zeros([K,F])
  for k in range(K): # iterate over all hidden states
    sum_of_weights = np.sum(gamma[:,k,:])
    weighted_sum_mean = 0
    weighted_sum_var = np.zeros([F,F])
    for n in range(N):
      weighted_sum_mean += gamma[n,k,:].reshape(1,-1).dot(x[n])
    new_mean[k,:] = weighted_sum_mean/sum_of_weights

    for n in range(N):
      T = x[n].shape[0]
      for t in range(T):
        centered = (x[n][t,:] - new_mean[k,:]).T
        weighted_sum_var += gamma[n,k,t]*(centered.dot(centered.T))
    
    # for now we just care abou the diagnal elements
    new_var[k,:] = (weighted_sum_var/sum_of_weights).diagonal()

  # --- update A ---
  # A is a K*K matrix (not symmertric)
  new_A = np.zeros(A.shape)
  for i in range(K):
    for j in range(K):
      e_i_to_j = 0 # the expected probability of transition from state i to j
      e_i_to_all = 0  # the expected number of transitions from state i

      for n in range(N):
        for t in range(T-1):
          b_j_ot1 = cal_b(x[n][t+1,:],B_mean[j,:],B_var[j,:])
          e_i_to_j += alpha[n][i,t]*beta[n][i,t+1]*A[i,j]*b_j_ot1
          e_i_to_all += alpha[n][i,t]*beta[n][i,t]

      new_A[i,j] = e_i_to_j/e_i_to_all

  return new_A, new_mean, new_var  # new_mean: K*F; new_var: K*F*F 


if __name__ == '__main__':
  main()











