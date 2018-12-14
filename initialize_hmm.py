# Functions for loading the data and scaling it
import os
import numpy as np 
from sklearn import preprocessing

def load_data():
  # load the training and testing data 
  # Change the directory to wherever the folder is located
      
  # "subject_train.txt" shows which subject number did the acts in "X_train"
  # ^ the rows of "subject_train" match up with the rows of "X_train"
  # ^ "y_test" gives the movement of each row from 1-6. NOTE: movements of 1 person can change during experiment
  # ^ 1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS, 4 SITTING, 5 STANDING, 6 LAYING
  
  os.chdir("/Users/SARL/Downloads/UCI HAR Dataset")
  f_x = open("train/X_train.txt", 'r')
  f_y = open("train/y_train.txt", 'r')
  f_s = open("train/subject_train.txt", 'r')
  x = f_x.read().split(' ')
  y = f_y.read().split('\n')
  s = f_s.read().split('\n')
  f_x.close()
  f_y.close()
  f_s.close()
  indices = [i for i, a in enumerate(x) if a != '']
  x_train = [float(x[j]) for j in indices]
  x_train = np.reshape(x_train, (7352, 561))
  y_train = np.array(y)
  y_train = y_train[0:7352].astype(int)
  s_train = np.array(s)
  s_train = s_train[0:7352].astype(int)
  
  f_x = open("test/X_test.txt", 'r')
  f_y = open("test/y_test.txt", 'r')
  f_s = open("test/subject_test.txt", 'r')
  x = f_x.read().split(' ')
  y = f_y.read().split('\n')
  s = f_s.read().split('\n')
  f_x.close()
  f_y.close()
  f_s.close()
  indices = [i for i, a in enumerate(x) if a != '']
  x_test = [float(x[j]) for j in indices]
  x_test = np.reshape(x_test, (2947, 561))
  y_test = np.array(y)
  y_test = y_test[0:2947].astype(int)
  s_test = np.array(s)
  s_test = s_test[0:2947].astype(int)
  return x_train, y_train, s_train, x_test, y_test, s_test

def init_par(H):
  # Labels: Walking = W, Walking Upstairs = WU, Walking Downstairs = WD, Sitting = SI, Standing = ST, Laying = LY
  # number of hidden states H 
  
  # initial prior probability distribution
  pi = (1.0/H)*np.ones(H)
  
  # initial transition probability matrix
  # A(i,j) is the probability that the hidden variable transititions from state i, to state j at some time t: P(S_t = j | S_(t-1) = i)
  A = (1.0/H) * np.ones((H,H))
  return A, pi

def segment_data(y):
  # Segment the activity sequences by taking the labels, and getting 
  # the starting and ending indices of a sequence
  cur_label = y[0] # current label
  start = 0
  activity_indices = []
  for i in range(0, len(y)):
    if y[i] != cur_label:
      end = i
      activity_indices.append([cur_label, start, end])
      start = end
      cur_label = y[i]
  # add in the last sequence
  activity_indices.append([cur_label, end, len(y)])
  return np.array(activity_indices)

def standardize_data(x):
  z = preprocessing.scale(x)
  return z

