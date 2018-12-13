### load data for messing around 

import hmm
import initialize_hmm
import log_FB_seq
import numpy as np

# write a general function that trains an HMM 
def train_model(s, H, K, d, n_mixture, n_iter):
    """ Trains an HMM with parameters s, H, K, d"""
    # s is a list of the labels to be trained on 
    # H is the number of hidden states to include in the model 
    # K is the length of the activity sequences 
    # d is a list of the features to be used for training the model
    # n_mixture is the number of Gaussians mixtures
    
    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()

    # standardize it (get z-scores)
    x_train = initialize_hmm.standardize_data(x_train) 
    
    # get the indices of each activity sequence 
    activity_train = initialize_hmm.segment_data(y_train)  
    
    # get the required segments of activities 
    all_segments = []
    for i in s:
        segment = hmm.all_sequences(x_train,i, activity_train)
        all_segments = all_segments + segment
    x = all_segments
    
    # initialize the transition matrix, the prior probabilities, and the Gaussians
    A, pi = initialize_hmm.init_par(H)
    w = (1.0/n_mixture) * np.ones((H,n_mixture))
    pi = np.asarray(pi)
    kmeans, B_mean, B_var = hmm.initialize_GaussianMixture(x, H, n_mixture)
    B_mean = B_mean[:,d,:]
    B_var  = B_var[:,d,:]
    for j in range(n_iter):
        print("Iteration {}".format(j))
        A, B_mean, B_var, pi, w, gamma_mat, gammaGMM_mat = log_FB_seq.forward_backward_algorithm(x, A, B_mean, B_var, pi, w, H, K, d)
    
    return A, B_mean, B_var, pi, w, gamma_mat, gammaGMM_mat

def predict_stage1(A_stat, B_mean_stat, B_var_stat, pi_stat, w_stat, A_mov, B_mean_mov, B_var_mov, pi_mov, w_mov, d, H, K):     
    """ Function for generating predictions from the first stage (moving or 
    stationary) model"""
    
    # get the appropriate y labels
    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()

    # get the indices of each activity sequence 
    activity_train = initialize_hmm.segment_data(y_train) 
    
    all_segments = []
    for i in range(1,7):
        segment = hmm.all_sequences(x_train,i, activity_train)
        all_segments = all_segments + segment
    x = all_segments
    
    E = len(x)
    L_mov = np.zeros((E,))
    L_stat = np.zeros((E,))
    y_pred = []
    for e in range(E):
        B_stat = hmm.cal_b_matrix_GMM(x[e][:,d], B_mean_stat, B_var_stat, w_stat, H, K)
        B_mov  = hmm.cal_b_matrix_GMM(x[e][:,d], B_mean_mov, B_var_mov, w_mov, H, K)
        
        alpha_stat = log_FB_seq.forward_step(A_stat, B_stat, pi_stat, w_stat, H, K)
        alpha_mov  = log_FB_seq.forward_step(A_mov, B_mov, pi_mov, w_mov, H, K)
        
        L_stat[e] = np.sum(alpha_stat[:,-1])
        L_mov[e] = np.sum(alpha_mov[:,-1])
        
        if L_stat[e] > L_mov[e]:
            y_pred.append(0)
        else:
            y_pred.append(1)
            
    return y_pred, L_stat, L_mov

def predict_stage2(s, A, B_mean, B_var, pi, d, H, K):
    """ Function for generating predictions from the second stage models"""
    # A should be a list of the transition matrices in order of either 1,2,3 or 4,5,6
    # same with B_mean, B_var, and pi
    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()

    # standardize it (get z-scores)
    x_test = initialize_hmm.standardize_data(x_test) 
    
    # get the indices of each activity sequence 
    activity_test = initialize_hmm.segment_data(y_train)  
    
    all_segments = []
    for i in s:
        segment = hmm.all_sequences(x_test,i, activity_test)
        all_segments = all_segments + segment
    x = all_segments
    
    E = len(x)
    L1 = np.zeros((E,))
    L2 = np.zeros((E,))
    L3 = np.zeros((E,))
    y_pred = []
    for e in range(E):
        B0  = hmm.cal_b_matrix(x[e][:,d], B_mean[0], B_var[0], H, K)
        B1  = hmm.cal_b_matrix(x[e][:,d], B_mean[1], B_var[1], H, K)
        B2  = hmm.cal_b_matrix(x[e][:,d], B_mean[2], B_var[2], H, K)
        
        alpha1 = log_FB_seq.forward_step(A[0], B0, pi[0], H, K)
        alpha2 = log_FB_seq.forward_step(A[1], B1, pi[1], H, K)
        alpha3 = log_FB_seq.forward_step(A[2], B2, pi[2], H, K)
        
        L1[e] = np.sum(alpha1[:,-1])
        L2[e] = np.sum(alpha2[:,-1])
        L3[e] = np.sum(alpha3[:,-1])
        
        current_L = np.array([L1[e], L2[e], L3[e]])
        label = [i for i, a in enumerate(current_L) if a == max(current_L)]
        y_pred.append(s[label[0]])
    return y_pred, L1, L2, L3 

def compute_error_stage1(y_pred):
    # get the appropriate y labels
    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()

    # get the indices of each activity sequence 
    activity_train = initialize_hmm.segment_data(y_train)  
    
    # get the required segments of activities 
    y_labels = []
    for i in range(1,7):
        segment = hmm.all_sequences(x_train,i, activity_train)
        y_labels = y_labels + (i*np.ones((len(segment)))).tolist()
    
    # compute error 
    error = 0.0
    E = len(y_labels)
    for e in range(E):
        if y_labels[e] == 1 or y_labels[e] == 2 or y_labels[e] == 3:
            current_label = 1
        else:
            current_label = 0
        
        if y_pred[e] != current_label:
            error+= 1
    error = error/E
    print("Testing error rate for Stage 1 is {}.".format(error))
    return error 

def compute_error_stage2(s, y_pred):
    # get the appropriate y labels
    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()

    # get the indices of each activity sequence 
    activity_test = initialize_hmm.segment_data(y_test)  
    
    # get the required segments of activities 
    y_labels = []
    for i in s:
        segment = hmm.all_sequences(x_test,i, activity_test)
        y_labels = y_labels + (i*np.ones((len(segment)))).tolist()
    
    # compute error 
    error = 0.0
    E = len(y_labels)
    for e in range(E):
        if y_pred[e] != y_labels[e]:
            error+= 1
    error = error/E
    print("Error rate for Stage 2 is {}.".format(error))
    return error
  
#%% Stage 1 models
    
d = [9, 6, 19, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 99]
H = 2
K = 7
n_mixture = 1
n_iter = 3
A_stat, B_mean_stat, B_var_stat, pi_stat, w_stat, gamma1, gammaGMM1 = train_model([4,5,6], H, K, d, n_mixture, n_iter)
A_mov, B_mean_mov, B_var_mov, pi_mov, w_mov, gamma2, gammaGMM2 = train_model([1,2,3], H, K, d, n_mixture, n_iter)
y_pred, L_stat, L_mov = predict_stage1(A_stat, B_mean_stat, B_var_stat, pi_stat, w_stat, A_mov, B_mean_mov, B_var_mov, pi_mov, w_mov, d, H, K)
error = compute_error_stage1(y_pred)

#%% Stage 2 models 

#d_mov = [37, 53, 65, 66, 67, 68, 69, 70, 71,73, 74,75, 76, 209, 222, 302 ,310]
#d_mov = range(561)
#A_W,  B_mean_W,  B_var_W,  pi_W  = train_model([1], 3, 7, d_mov, 2)
#A_WU, B_mean_WU, B_var_WU, pi_WU = train_model([2], 3, 7, d_mov, 2)
#A_WD, B_mean_WD, B_var_WD, pi_WD = train_model([3], 3, 7, d_mov, 2)

#d_stat = [40, 41, 42, 49, 50, 51, 52, 53, 54, 56, 77, 558, 559, 560]
#A_SI, B_mean_SI, B_var_SI, pi_SI = train_model([4], 3, 7, d_stat, 6)
#A_ST, B_mean_ST, B_var_ST, pi_ST = train_model([5], 3, 7, d_stat, 6)
#A_LY, B_mean_LY, B_var_LY, pi_LY = train_model([6], 3, 7, d_stat, 6)

#y_pred = predict_stage2([1,2,3], [A_W, A_WU, A_WD], [B_mean_W, B_mean_WU, B_mean_WD], [B_var_W, B_var_WU, B_var_WD], [pi_W, pi_WU, pi_WD], d_mov, 3, 7)
#error  = compute_error_stage2([1,2,3],y_pred)




#%%
#x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()
#
## standardize it (get z-scores)
#x_train = initialize_hmm.standardize_data(x_train) 
#
## get the indices of each activity sequence 
#activity_train = initialize_hmm.segment_data(y_train)  
#
## get the stationary segments 
#segments1 = hmm.all_sequences(x_train,1, activity_train)
#segments2 = hmm.all_sequences(x_train,2, activity_train)
#segments3 = hmm.all_sequences(x_train,3, activity_train)
#segments4 = hmm.all_sequences(x_train,4, activity_train)
#segments5 = hmm.all_sequences(x_train,5, activity_train)
#segments6 = hmm.all_sequences(x_train,6, activity_train)
#
#x_train_stat = segments4 + segments5 + segments6
#x_train_mov  = segments1 + segments2 + segments3
#
#H = 2
#K = 7
#d = [9, 6, 19, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 99]
##d = range(561)
#
## stationary HMM
#A_stat, pi_stat = initialize_hmm.init_par(H) # x_train, y_train not used in the function
#pi_stat = np.asarray(pi_stat)
#kmeans, B_mean_stat, B_var_stat = hmm.initialize_GMM(x_train_stat, H)
#B_mean_stat = B_mean_stat[:,d]
#B_var_stat  = B_var_stat[:,d]
#for i in range(3):
#    print("Iteration {}".format(i))
#    A_stat, B_mean_stat, B_var_stat, pi_stat = log_FB_seq.forward_backward_algorithm(x_train_stat, A_stat, B_mean_stat, B_var_stat, pi_stat, H, K, d)
#
## moving HMM
#A_mov, pi_mov = initialize_hmm.init_par(H) # x_train, y_train not used in the function
#pi_mov = np.asarray(pi_mov)
#kmeans, B_mean_mov, B_var_mov = hmm.initialize_GMM(x_train_mov, H)
#B_mean_mov = B_mean_mov[:,d]
#B_var_mov = B_var_mov[:,d]
#for i in range(3):
#    print("Iteration {}".format(i))
#    A_mov, B_mean_mov, B_var_mov, pi_mov = log_FB_seq.forward_backward_algorithm(x_train_mov, A_mov, B_mean_mov, B_var_mov, pi_mov, H, K, d)
#
##%% Compute misclassification rate on training error 
#    
## compute likelihoods
#all_train = segments1 + segments2 + segments3 + segments4 + segments5 + segments6
#E = len(all_train)
#L_mov = np.zeros((E,))
#L_stat = np.zeros((E,))
#y_pred = []
#for e in range(E):
#    B_stat = hmm.cal_b_matrix(all_train[e][:,d], B_mean_stat, B_var_stat, H, K)
#    B_mov  = hmm.cal_b_matrix(all_train[e][:,d], B_mean_mov, B_var_mov, H, K)
#    
#    alpha_stat = log_FB_seq.forward_step(A_stat, B_stat, pi_stat, H, K)
#    alpha_mov  = log_FB_seq.forward_step(A_mov, B_mov, pi_mov, H, K)
#    
#    L_stat[e] = np.sum(alpha_stat[:,-1])
#    L_mov[e] = np.sum(alpha_mov[:,-1])
#    
#    if L_stat[e] > L_mov[e]:
#        y_pred.append(0)
#    else:
#        y_pred.append(1)
#
#
## compute error 
#error = 0.0
#for e in range(E):
#    if y_train[e] == 1 or y_train[e] == 2 or y_train[e] == 3:
#        current_label = 1
#    else:
#        current_label = 0
#    if y_pred[e] != current_label:
#        error+= 1
#error = error/E
#
#print("Training error rate for Stage 1 is {}.".format(error))
#
#
#
##%% Compute misclassification rate on training error 
## pre-process test data
#x_test = initialize_hmm.standardize_data(x_test)
#activity_test = initialize_hmm.segment_data(y_test)
#
## get the stationary segments 
#segments1_test = hmm.all_sequences(x_test,1, activity_test)
#segments2_test = hmm.all_sequences(x_test,2, activity_test)
#segments3_test = hmm.all_sequences(x_test,3, activity_test)
#segments4_test = hmm.all_sequences(x_test,4, activity_test)
#segments5_test = hmm.all_sequences(x_test,5, activity_test)
#segments6_test = hmm.all_sequences(x_test,6, activity_test)
#
#x_test_stat = segments4_test + segments5_test + segments6_test
#x_test_mov  = segments1_test + segments2_test + segments3_test
#
## compute likelihoods
#all_test = x_test_mov + x_test_stat
#E = len(all_test)
#L_mov = np.zeros((E,))
#L_stat = np.zeros((E,))
#y_pred = []
#for e in range(E):
#    B_stat = hmm.cal_b_matrix(all_test[e][:,d], B_mean_stat, B_var_stat, H, K)
#    B_mov  = hmm.cal_b_matrix(all_test[e][:,d], B_mean_mov, B_var_mov, H, K)
#    
#    alpha_stat = log_FB_seq.forward_step(A_stat, B_stat, pi_stat, H, K)
#    alpha_mov  = log_FB_seq.forward_step(A_mov, B_mov, pi_mov, H, K)
#    
#    L_stat[e] = np.sum(alpha_stat[:,-1])
#    L_mov[e] = np.sum(alpha_mov[:,-1])
#    
#    if L_stat[e] > L_mov[e]:
#        y_pred.append(0)
#    else:
#        y_pred.append(1)
#
#
## compute error 
#error = 0
#for e in range(E):
#    if y_test[e] == 1 or y_test[e] == 2 or y_test[e] == 3:
#        current_label = 1
#    else:
#        current_label = 0
#    
#    if y_pred[e] != current_label:
#        error+= 1
#error = error/E
#
#print("Testing error rate for Stage 1 is {}.".format(error))

#%% Train models for Walking, Walking Upstairs, Walking Downstairs 




