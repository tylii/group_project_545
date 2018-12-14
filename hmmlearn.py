#pred_y = hmm(x_train,y_train,x_test,top_features_l1, top_features_l2_123,top_features_l2_456)
import testing
import initialize_hmm
import hmm
import numpy as np

def hmmlearn(x_train, y_train, x_test, y_test, top_features_l1, top_features_l2_l123, top_features_l2_l456, n_mixture = 1, n_iter = 3):
    """ Overarching two-stage HMM classifier """
        
    ### Build model for layer 1 
    # parameters 
    d = top_features_l1
    H1 = 2
    K = 7
    
    A_stat, B_mean_stat, B_var_stat, pi_stat, w_stat = testing.train_model([4,5,6], x_train, y_train, H1, K, d, n_mixture, n_iter)
    A_mov, B_mean_mov, B_var_mov, pi_mov, w_mov = testing.train_model([1,2,3], x_train, y_train, H1, K, d, n_mixture, n_iter)
    
    ### Build model for layer 2
    # parameters
    d123 = top_features_l2_l123
    d456 = top_features_l2_l456
    H2 = 3
    
    A_W,  B_mean_W,  B_var_W,  pi_W,  w_W  = testing.train_model([1], x_train, y_train, H2, K, d123, n_mixture, n_iter)
    A_WU, B_mean_WU, B_var_WU, pi_WU, w_WU = testing.train_model([2], x_train, y_train, H2, K, d123, n_mixture, n_iter)
    A_WD, B_mean_WD, B_var_WD, pi_WD, w_WD = testing.train_model([3], x_train, y_train, H2, K, d123, n_mixture, n_iter)
    
    A_SI, B_mean_SI, B_var_SI, pi_SI, w_SI = testing.train_model([4], x_train, y_train, H2, K, d456, n_mixture, n_iter)
    A_ST, B_mean_ST, B_var_ST, pi_ST, w_ST = testing.train_model([5], x_train, y_train, H2, K, d456, n_mixture, n_iter)
    A_LY, B_mean_LY, B_var_LY, pi_LY, w_LY = testing.train_model([6], x_train, y_train, H2, K, d456, n_mixture, n_iter)
    
    # pre-process the testing data 
    activity_test = initialize_hmm.segment_data(y_test)

    # make predictions for layer 1
    y_pred = []
    E = len(activity_test)
    for i in range(E):
        print(i)
        segment, valid = hmm.activity_sequence(i, activity_test, x_test, K)        
        
        if valid:    
            prediction_stage1 = testing.predict_stage1_2(segment, A_stat, B_mean_stat, B_var_stat, pi_stat, w_stat, A_mov, B_mean_mov, B_var_mov, pi_mov, w_mov, d, H1, K)
            
            if prediction_stage1 == 1:
                prediction_stage2 = testing.predict_stage2_2(segment, [1,2,3], [A_W, A_WU, A_WD], [B_mean_W, B_mean_WU, B_mean_WD], [B_var_W, B_var_WU, B_var_WD], [pi_W, pi_WU, pi_WD], [w_W, w_WU, w_WD], d123, H2, K)
            else:
                prediction_stage2 = testing.predict_stage2_2(segment, [4,5,6], [A_SI, A_ST, A_LY], [B_mean_SI, B_mean_ST, B_mean_LY], [B_var_SI, B_var_ST, B_var_LY], [pi_SI, pi_ST, pi_LY], [w_SI, w_ST, w_LY], d456, H2, K)
        else:
            prediction_stage2 = np.nan
        
        # get back the original size of the y_test
        og_length = activity_test[i,2]-activity_test[i,1]
        y_og      = [prediction_stage2 for a in range(int(og_length))]
        y_pred += y_og
    
    # remove the nan's due to invalid sequence lengths 
    nan_indices = np.where(np.isnan(y_pred))
    y_pred = np.delete(y_pred, nan_indices[0])
    y_test = np.delete(y_test, nan_indices[0])
    return y_pred, y_test
    
    
    
    
    
    
    
    
    
    
    
    