## Random forest feature selection implementation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

def create_RF(x, y):
	# Create a classifer and return a 561 length vector of the feature importances
	clf = RandomForestClassifier(n_estimators = 1000)
	clf.fit(x, y) 
	return clf

def compute_average_importance(i, x, y, n = 15):
    all_importances = []
    for i in range(i):
        # i is the number of iterations 
        forest = create_RF(x, y)
        importances = forest.feature_importances_
#        importances = permutation_importances(forest, x, y, score)
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(n), importances[indices[0:n]], color = "r", yerr = std[indices[0:n]], align = "center")
        plt.xticks(range(n), indices[0:n])
        plt.xlim([-1, n])
        plt.show()
    #    plt.bar(range(x.shape[1]), importances[indices],
    #           color="r", yerr=std[indices], align="center")
    #    plt.xticks(range(x.shape[1]), indices)
    #    plt.xlim([-1, x.shape[1]])
    #    plt.show()
        all_importances.append(importances)
    return np.asarray(all_importances), forest

def score(model, x_train, y_train, k = 3):
    cvscore = cross_val_score(
            model,  # which model to use
            x_train, y_train,  # what training data to split up
            cv=k)  # number of folds/chunks
    return np.mean(cvscore)

def permutation_importances(rf, X_train, y_train, metric):
    # Method for computing feature importance (not the default)
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in range(X_train.shape[1]):
        save = X_train[:,col].copy()
        X_train[:,col] = np.random.permutation(X_train[:,col])
        m = metric(rf, X_train, y_train)
        X_train[:,col] = save
        imp.append(baseline - m)
    return np.array(imp)

importances, forest = compute_average_importance(1, x_train, y_boop)
