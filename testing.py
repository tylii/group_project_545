import hmm
import initialize_hmm
import log_FB_seq
import numpy as np

# write a general function that trains an HMM 
def train_model(s, x_train, y_train, H, K, d, n_mixture, n_iter):
    """ Trains an HMM with parameters s, H, K, d"""
    # s is a list of the labels to be trained on 
    # H is the number of hidden states to include in the model 
    # K is the length of the activity sequences 
    # d is a list of the features to be used for training the model
    # n_mixture is the number of Gaussians mixtures

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
        A, B_mean, B_var, pi, w = log_FB_seq.forward_backward_algorithm(x, A, B_mean, B_var, pi, w, H, K, d)
    
    return A, B_mean, B_var, pi, w

def predict_stage1_2(x, A_stat, B_mean_stat, B_var_stat, pi_stat, w_stat, A_mov, B_mean_mov, B_var_mov, pi_mov, w_mov, d, H, K):     
    """ Function for generating single predictions from the first stage (moving or 
    stationary) model for a sequence"""
    
    B_stat = hmm.cal_b_matrix_GMM(x[:,d], B_mean_stat, B_var_stat, w_stat, H, K)
    B_mov  = hmm.cal_b_matrix_GMM(x[:,d], B_mean_mov, B_var_mov, w_mov, H, K)
    
    alpha_stat = log_FB_seq.forward_step(A_stat, B_stat, pi_stat, w_stat, H, K)
    alpha_mov  = log_FB_seq.forward_step(A_mov, B_mov, pi_mov, w_mov, H, K)
    
    L_stat = np.sum(alpha_stat[:,-1])
    L_mov = np.sum(alpha_mov[:,-1])
    
    if L_stat > L_mov:
        y_pred = 0
    else:
        y_pred = 1
    return y_pred
    
    
def predict_stage1(x_predict, y, A_stat, B_mean_stat, B_var_stat, pi_stat, w_stat, A_mov, B_mean_mov, B_var_mov, pi_mov, w_mov, d, H, K):     
    """ Function for generating predictions from the first stage (moving or 
    stationary) model"""

    # get the indices of each activity sequence 
    activity = initialize_hmm.segment_data(y) 
    
    all_segments = []
    for i in range(1,7):
        segment = hmm.all_sequences(x_predict,i, activity)
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

def predict_stage2_2(x, s, A, B_mean, B_var, pi, w, d, H, K):
    """ new predict function that takes in one sequence"""
    B0  = hmm.cal_b_matrix_GMM(x[:,d], B_mean[0], B_var[0], w[0], H, K)
    B1  = hmm.cal_b_matrix_GMM(x[:,d], B_mean[1], B_var[1], w[1], H, K)
    B2  = hmm.cal_b_matrix_GMM(x[:,d], B_mean[2], B_var[2], w[2], H, K)
    
    alpha1 = log_FB_seq.forward_step(A[0], B0, pi[0], w[0], H, K)
    alpha2 = log_FB_seq.forward_step(A[1], B1, pi[1], w[1], H, K)
    alpha3 = log_FB_seq.forward_step(A[2], B2, pi[2], w[2], H, K)
    
    L1 = np.sum(alpha1[:,-1])
    L2 = np.sum(alpha2[:,-1])
    L3 = np.sum(alpha3[:,-1])
    
    current_L = np.array([L1, L2, L3])
    label = [i for i, a in enumerate(current_L) if a == max(current_L)]
    y_pred = s[label[0]]
    return y_pred


def predict_stage2(x_pred, y, s, A, B_mean, B_var, pi, w, d, H, K):
    """ Function for generating predictions from the second stage models"""
    # A should be a list of the transition matrices in order of either 1,2,3 or 4,5,6
    # same with B_mean, B_var, and pi

    activity = initialize_hmm.segment_data(y)  
    
    all_segments = []
    for i in s:
        segment = hmm.all_sequences(x_pred,i, activity)
        all_segments = all_segments + segment
    x = all_segments 
        
    E = len(x)
    L1 = np.zeros((E,))
    L2 = np.zeros((E,))
    L3 = np.zeros((E,))
    y_pred = []
    for e in range(E):
        print(e)
        B0  = hmm.cal_b_matrix_GMM(x[e][:,d], B_mean[0], B_var[0], w[0], H, K)
        B1  = hmm.cal_b_matrix_GMM(x[e][:,d], B_mean[1], B_var[1], w[1], H, K)
        B2  = hmm.cal_b_matrix_GMM(x[e][:,d], B_mean[2], B_var[2], w[2], H, K)
        
        alpha1 = log_FB_seq.forward_step(A[0], B0, pi[0], w[0], H, K)
        alpha2 = log_FB_seq.forward_step(A[1], B1, pi[1], w[1], H, K)
        alpha3 = log_FB_seq.forward_step(A[2], B2, pi[2], w[2], H, K)
        
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
    return error, y_labels

def compute_error_stage2(s, y_pred, data_type):
    # get the appropriate y labels
    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()

    # get the indices of each activity sequence 
    activity_train = initialize_hmm.segment_data(y_train)  
    activity_test  = initialize_hmm.segment_data(y_test)
    
    # get the required segments of activities 
    if data_type == "train":
        y_labels = []
        for i in s:
            segment = hmm.all_sequences(x_train,i, activity_train)
            y_labels = y_labels + (i*np.ones((len(segment)))).tolist()
    elif data_type == "test":
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
    print("{} Error rate for Stage 2 is {}.".format(data_type, error))
    return error, y_labels





