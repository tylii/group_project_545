### Script for running training the models 

import hmm
import viterbi

n_iterations = 5
# stationary HMM
A_stat, B_mean_stat, B_var_stat, pi_stat , alpha_stat, beta_stat = hmm.hmm_train(7, 2, n_iterations, "stationary")

# moving HMM
A_mov, B_mean_mov, B_var_mov, pi_mov, alpha_mov, beta_mov = hmm.hmm_train(7, 2, n_iterations, "moving")

# Compute predictions using Viterbi
# load testing data 
#x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()


def predict(segments):
    prob_stat_mat = []
    prob_mov_mat = []
    for i in range(len(segments)):
        prob_stat = viterbi.compute_viterbi(segments[i], B_mean_stat, B_var_stat, A_stat, pi_stat, 2, 7)
        prob_mov = viterbi.compute_viterbi(segments[i], B_mean_mov, B_var_mov, A_mov, pi_mov, 2, 7)
        prob_stat_mat.append(prob_stat)
        prob_mov_mat.append(prob_mov)
    return prob_stat_mat, prob_mov_mat

# Compute test error 


prob_stat_mat, prob_mov_mat = predict(segments)