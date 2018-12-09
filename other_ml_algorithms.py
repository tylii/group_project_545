  # load the data
 
import initialize_hmm
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from hmm import all_sequences
import numpy as np 
import matplotlib.pyplot as plt


def get_data(features, ACT1, ACT2):
    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()
    x_train = initialize_hmm.standardize_data(x_train) 
    x,y = get_data_for_these_activities(x_train,y_train,ACT1,ACT2,features)
    return x,y


def feature_selection_RF(ACT1, ACT2):
    x,y = get_data(range(561),ACT1, ACT2)
    important_features = rf_classifier(x,y)
    return important_features
    # for i in range(N_f):
    #    rf_classifier(x,y)

 
def try_other_classifier(features, ACT1, ACT2):
    important_features = feature_selection_RF(ACT1, ACT2)
    x,y = get_data(important_features,ACT1, ACT2)
    svm_classifier(x,y)
    rf_classifier(x,y)

def svm_classifier(x,y):
    n_train = int(len(y)*0.8)
    x_train = x[0:n_train,:]
    y_train = y[0:n_train]

    x_test = x[n_train:len(y),:]
    y_test = y[n_train:len(y)]

    
    clf = svm.SVC()
    clf.fit(x_train, y_train)  
    pred_Y = clf.predict(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_Y, pos_label=2)

    print(pred_Y)
    print(y_test)

    plt.plot(y_test,pred_Y)
    plt.show()
    auc = metrics.auc(fpr, tpr)
    print(auc)

def get_data_for_these_activities(x_train,y_train,ACT1,ACT2,features):
    # feature is the index of features e.g. [0, 14, 99]
    len_data = sum(1 for x in y_train if x in ACT1 or x in ACT2)
    new_y = np.zeros([len_data])
    new_x = np.zeros([len_data,len(features)])
    j=0
    for i in range(x_train.shape[0]):
        if y_train[i] in ACT1:
            new_y[j] = 0
            new_x[j,:] = x_train[i,features]
            j+=1
        if y_train[i] in ACT2:
            new_y[j] = 1
            new_x[j,:] = x_train[i,features]
            j+=1
    
    return new_x,new_y

def rf_classifier(x,y):
    forest = RandomForestClassifier(n_estimators=2000,max_depth=2,random_state=0)
    forest.fit(x,y)
    # print(sorted(forest.feature_importances_,reverse=True))
    importances = forest.feature_importances_ 
    # importance stores the importance of the firest feature, second feature,... (it is nor sorted by value)    
    indices = np.argsort(importances)[::-1] # return the reversed indices that sort the importance.
    # print(min(indices))

  # Print the feature ranking
  # print("Feature ranking:")
  # for f in range(x.shape[1]):
  #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

  # Print the feature ranking of top 20
    print("Feature ranking:")
    # the feature index start from 0
    for f in range(20):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    return indices # it store the index of the most important feature, second-most important feature, and so on

if __name__ == '__main__':
    parameters = [[9,83,86],[1,2,3], [4,5,6]]
    # try_other_classifier([9,83,86],[1,2,3], [4,5,6])
    feature_selection_RF([1],[2])
