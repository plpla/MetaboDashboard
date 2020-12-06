import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier

LEARN_CONFIG={
    
    "CV_folds": 3, # Numbre of cross-validation fold. This is performed on each split
    "Algos":{ # Each algo will be optimized on each split. CV is used to find the optimal parameters combination from the param grid.
        "RandomForest":{
            "function": RandomForestClassifier,
            "ParamGrid": {
                "n_estimators": [30, 100]
            }
        },
    }
}








