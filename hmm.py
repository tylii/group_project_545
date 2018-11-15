## Implementing Hidden Markov Model using Baum-Welch algorithm
# Load the data 

# "subject_train.txt" shows which subject number did the acts in "X_train"
# ^ the rows of "subject_train" match up with the rows of "X_train"
# ^ "y_test" gives the movement of each row from 1-6. NOTE: movements of 1 person can change during experiment
# ^ 1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS, 4 SITTING, 5 STANDING, 6 LAYING

 
import os
import numpy as np 

def main():
  # load training X training and y training data 
  x,y = load_data()
  #-----------------------------------------#
	## Baum-Welch algorithm 
  ## Step 1: Initialize all Gaussian distributions with the mean and variance along the whole dataset.
  A,B,pi = init_par()
  ## Step 2: Calculate the forward and backward probabilities for all states j and times t.
  for ? :
    forward_step()
    backward_step()
    update_par()
  
  
def load_data():
	os.chdir("/Users/SARL/Downloads/UCI HAR Dataset/train")
	f_x = open("X_train.txt", 'r')
	f_y = open("y_train.txt", 'r')
  f_s = open("subject_train.txt", 'r')
	x = f_x.read().split(' ')
	y = f_y.read().split('\n')
  s = f_s.read().split('\n')
	f_x.close()
	f_y.close()
  f_s.close()
  indices = [i for i, a in enumerate(x) if a != '']
	x_train = [x[j] for j in indices]
	x_train = np.reshape(x_train, (7352, 561))
	y_train = np.array(y)
	y_train = y_train[0:7352]
  s_train = np.array(s)
  s_train = s_train[0:7352]
  return x_train,y_train


def init_par()
	# Labels: Walking = W, Walking Upstairs = WU, Walking Downstairs = WD, Sitting = SI, Standing = ST, Laying = LY
	# # features = m , # training samples = n, # of hidden states = K, # time points = T
	# initial prior probability distribution 
	pi_W, pi_WU, pi_WD, pi_SI, pi_ST, pi_LY = [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0]
	
	# initial transition probability matrix
	# A(i,j) is the probability that the hidden variable transititions from state i, to state j at some time t: P(S_t = j | S_(t-1) = i)
	A_easy = (1.0/6.0) * np.ones((6,6))
	
	# initial emission probabilities matrix 
  # B(i,j) is the probabilities that the state i, will emmit output variable j.
  # get the indices for each activity 
  ind_W  = [i for i, a in enumerate(y) if a == 1]
  ind_WU = [i for i, a in enumerate(y) if a == 2]
  ind_WD = [i for i, a in enumerate(y) if a == 3]
  ind_SI = [i for i, a in enumerate(y) if a == 4]
  ind_ST = [i for i, a in enumerate(y) if a == 5]
  ind_LY = [i for i, a in enumerate(y) if a == 6]
  
  x_W  = [x_train[j] for j in ind_W]
  x_WU = [x_train[j] for j in ind_WU]
  x_WD = [x_train[j] for j in ind_WD]
  x_SI = [x_train[j] for j in ind_SI]
  x_ST = [x_train[j] for j in ind_ST]
  x_LY = [x_train[j] for j in ind_LY]
  
  avg_xW  = np.mean(x_W, axis = 0) 
  avg_xWU = np.mean(x_WU, axis = 0)
  avg_xWD = np.mean(x_WD, axis = 0)
  avg_xSI = np.mean(x_SI, axis = 0)
  avg_xST = np.mean(x_ST, axis = 0)
  avg_xLY = np.mean(x_LY, axis = 0)
  
  averages_x = [avg_xW, avg_xWU, avg_xWD, avg_xSI, avg_ST, avg_xLY]
  
  var_xW  = np.var(x_W, axis = 0) 
  var_xWU = np.var(x_WU, axis = 0)
  var_xWD = np.var(x_WD, axis = 0)
  var_xSI = np.var(x_SI, axis = 0)
  var_xST = np.var(x_ST, axis = 0)
  var_xLY = np.var(x_LY, axis = 0)
  
  variances_x = [var_xW, var_xWU, var_xWD, var_xSI, var_ST, var_xLY]
  A = A_easy
  B = 
          
  return(A,B,pi)
	# theta_init= [A, B, pi]
	
def forward_step(x,y,K,T,B):
	# calculate the forward probabilities
	# alpha(i,t) is the probability of seeing x_1, x_2,... x_t (observations) and being in state i at time t
	# size of alpha: K * T
  alpha = np.zeros[K,T]

	for i in range(K):
    alpha[i,0] = pi[i],B[i,x[i]-1] # need edit
    for t in range(1,T):
      alpha[i,t] = B[i,y_t+1]
      
  return alpha
       
def backward_step(x,y,A,B):


## Step 3: For each state j and time t, use the probability Lj(t) and the current observation vector Ot to update the accumulators for that state.

## Step 4: Use the final accumulator values to calculate new parameter values.

## Step 5: If the value of P = P(O/M) for this iteration is not higher than the value at the previous iteration then stop, repeat
## the above steps using the new re-estimated parameter
## values (from step 2) 

#-----------------------------------------#
if __name__ == '__main__':
  main()
