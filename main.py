### Script for running training the models 

import hmm

# stationary HMM
A_stat, B_mean_stat, B_var_stat, pi_stat , alpha_stat, beta_stat = hmm.hmm_train(7, 2, 5, "stationary")

# moving HMM
A_mov, B_mean_mov, B_var_mov, pi_mov, alpha_mov, beta_mov = hmm.hmm_train(7, 2, 5, "moving")

# Compute predictions using Viterbi