  # load the data
 
import initialize_hmm
from sklearn import svm, metrics, linear_model
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from hmm import all_sequences
import numpy as np 
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools


def try_other_classifier(n_features,n_features2_1,n_features2_2,Kfold):
    '''
    This function test other classifier, using K-fold cross validation.
    n_features: how many top features will be used
    ACT1 and ACT2: the two sets of activities
    Kfold: K-fold cross validation
    '''
    x,y = get_data([1,2,3,4,5,6])  # here y is like [3,1,4,4,5,2 ....] (not binary label)
    folds = split_folds_by_sequence(x,y,Kfold)

    svm_scores = np.zeros(len(folds))
    # rf_scores = np.zeros(len(folds))
    for i in range(len(folds)):
        x_train, y_train,x_test,y_test = get_train_test_from_folds(folds,i)
        # ----feature selection ------
        # layer 1 
        y_train_b = binary_label(y_train,[1,2,3],[4,5,6]) # y with binary values [0,1] 
        important_features = feature_selection_layer1(x_train, y_train_b)
        top_features_l1 = important_features[0:n_features]

        # layer 2
        top_features_l2_123 = feature_selection_layer2(x_train, y_train,[1,2,3],n_features2_1)
        top_features_l2_456 = feature_selection_layer2(x_train, y_train,[4,5,6],n_features2_2)

        # ------ HMM -----
        # e.g. pred_y = hmm(x_train,y_train,x_test,top_features_l1, top_features_l2_123,top_features_l2_456)
        # ------  SVM ----------
        pred_y = svm_classifier(x_train,y_train,x_test,top_features_l1, top_features_l2_123,top_features_l2_456)
        score = performance(pred_y,y_test)
        svm_scores[i] = score
        # ------ Random Forest -----
        # pred_y = rf_classifier(x_train, y_train_b,x_test)
        # score = performance(pred_y,y_test,y_test_b)
        # rf_scores[i] = score

    print('Performance of SVM: {}'.format(svm_scores))
    print(svm_scores)
    # print('Performance of RF: {}'.format(rf_scores))
    # print(rf_scores)



def get_data(ACT):
    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()
    x_train = initialize_hmm.standardize_data(x_train) 
    x,y = get_data_for_these_activities(x_train,y_train,ACT)
    return x,y


def feature_selection_layer1(x,y):
    '''
    This function conducts feature selection for 123 vs 456 classification
    '''
    forest = RandomForestClassifier(n_estimators=1000,max_depth=2,random_state=0)
    forest.fit(x,y)
    importances = forest.feature_importances_ 
    indices = np.argsort(importances)[::-1] # return the reversed indices that sort the importance.

    # Print the feature ranking of top 20
    # print("Feature ranking:")
    # # the feature index start from 0
    # for f in range(20):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    return indices

def feature_selection_layer2(x,y,ACT,n_features):
    '''
    This function conducts feature selection for wtihin 123/456 classification
    A set of features are firest selected for each pair (12,23,13), then these features are combined

    The output is a list of top features (not the same as feature_selection_layer1())
    '''
    all_features=[]
    for act in itertools.combinations(ACT, 2):
        x,y = get_data_for_these_activities(x,y,ACT)
        y_b = binary_label(y,[act[0]],[act[1]])
        features = feature_selection_layer1(x,y_b)
        all_features.append(features[0:n_features])
    all_features = np.concatenate(all_features,axis=0)
    all_features = np.unique(all_features)
    return all_features 


def feature_selection_RF(x,y):
    '''
    this function does feature selection
    the output stores the index of the most important feature, second-most important feature... so on
    '''
    forest = RandomForestClassifier(n_estimators=1000,max_depth=2,random_state=0)
    forest.fit(x,y)
    # print(sorted(forest.feature_importances_,reverse=True))
    importances = forest.feature_importances_ 
    # importance stores the importance of the firest feature, second feature,... (it is nor sorted by value)    
    indices = np.argsort(importances)[::-1] # return the reversed indices that sort the importance.

    # Print the feature ranking of top 20 (the feature index start from 0)
    print("Feature ranking:")
    for f in range(20):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    return indices


def split_folds_by_sequence(x,y,Kfold):
    np.random.seed(49)
    folds = defaultdict(lambda: defaultdict(lambda: []))
    last_y = y[0]
    last_g = np.random.randint(Kfold)
    for i in range(len(y)):
        if last_y != y[i]:
            last_g = np.random.randint(Kfold)
        folds[last_g]['x'].append(x[i,:])
        folds[last_g]['y'].append(y[i])
        last_y = y[i]
    return folds


def get_train_test_from_folds(folds,i):
    x_test = np.array(folds[i]['x'])
    y_test = np.array(folds[i]['y'])

    x_train = []
    y_train = []

    for j in [x for x in range(len(folds)) if x != i]:
        x_train.append(np.array(folds[j]['x']))
        y_train.append(np.array(folds[j]['y']))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    return x_train, y_train,x_test,y_test


def binary_label(y,ACT1,ACT2):
    y_b = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] in ACT1:
            y_b[i] = 1
        elif y[i] in ACT2:
            y_b[i] = 0
    return y_b


def svm_classifier(x_train,y_train,x_test,top_f_l1,top_f_l2_123,top_f_l2_456):

    #  ------ training ---------
    # layer 1
    y_train_b_l1 = binary_label(y_train,[1,2,3],[4,5,6])
    model_l1 = svm.SVC()
    model_l1.fit(x_train[:,top_f_l1], y_train_b_l1)  
    
    # layer 2 [1,2,3]
    x,y = get_data_for_these_activities(x_train,y_train,[1,2,3])
    model_l2_123 = svm.SVC()
    model_l2_123.fit(x[:,top_f_l2_123], y)  
    
    # layer 2 [4,5,6]
    x,y = get_data_for_these_activities(x_train,y_train,[4,5,6])
    model_l2_456 = svm.SVC()
    model_l2_456.fit(x[:,top_f_l2_456], y)  

    #  ------ testing ---------
    pred_Y_l1 = model_l1.predict(x_test[:,top_f_l1])
    pred_Y_l2_123 = model_l2_123.predict(x_test[:,top_f_l2_123])
    pred_Y_l2_456 = model_l2_456.predict(x_test[:,top_f_l2_456])

    pred_Y = get_final_prediction(pred_Y_l1,pred_Y_l2_123,pred_Y_l2_456)

    return pred_Y


def get_data_for_these_activities(x_train,y_train,ACT):
    '''
    output x and y for activities listed in ACT
    '''
    features = x_train.shape[1] # total number of features
    len_data = sum(1 for y in y_train if y in ACT) # the length of the output data
    new_y = np.zeros([len_data])
    new_x = np.zeros([len_data,features])
    j=0
    for i in range(x_train.shape[0]):
        if y_train[i] in ACT:
            new_y[j] = y_train[i]
            new_x[j,:] = x_train[i,:]
            j+=1
    return new_x,new_y

def rf_classifier(x_train, y_train_b,x_test):
    forest = RandomForestClassifier(n_estimators=2000,max_depth=2,random_state=0)
    forest.fit(x_train,y_train_b)
    y_pred = forest.predict(x_test)
    return y_pred


def get_final_prediction(pred_Y_l1,pred_Y_l2_123,pred_Y_l2_456):
    for i in range(len(pred_Y_l1)):
        if pred_Y_l1[i]==1:
            pred_Y_l1[i] = pred_Y_l2_123[i]
        else:
            pred_Y_l1[i] = pred_Y_l2_456[i]
    return pred_Y_l1


def performance(pred_y,y):
    '''
    In previous steps we are making predictions with respect to each time frame
    This function decides the predicted lable for a sequence using majority voting.
    '''
    tru_lab = []
    pred_lab = []

    last_y = y[0]

    tmp_score = [] # stores the predicted labels of the current sequence
    for i in range(len(y)):
        if last_y == y[i]:
            tmp_score.append(pred_y[i])
        else:
            pred_lab.append(max(set(tmp_score), key = tmp_score.count)) # majory voting
            tru_lab.append(y[i-1])
            tmp_score = []
        last_y = y[i]

    print(tru_lab)
    print(pred_lab)

    print(metrics.accuracy_score(tru_lab, pred_lab))
    return metrics.accuracy_score(tru_lab, pred_lab)


if __name__ == '__main__':
    parameters = [[9,83,86],[1,2,3], [4,5,6]]
    # try_other_classifier([9,83,86],[1,2,3], [4,5,6])
    # feature_selection_RF([1],[2])

    try_other_classifier(10,10,10,5)  
    
    #number of features for first layer, number of features for second layer 123,  number of features for second layer 456, K-fold cross validation
