import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

LEARN_CONFIG={
    "Nsplit": 100,
    #"Preprocessing": None #Could be [StandardScaler] if StandardScaler is imported. Not implemented yet...
    "UseNormalized": True, # Make sure it works if False. Is it only when using Progenesis input file?
    "CV_folds": 5,
    "Algos":{
        "DecisionTree":{
            "function": DecisionTreeClassifier,
            "ParamGrid": {
                "max_depth": [1, 2, 3, 4,],
                "min_samples_split": [2, 3, 4]
            }
        },
        "RandomForest":{
            "function": RandomForestClassifier,
            "ParamGrid": {
                "n_estimators": [1, 2, 4, 10, 30, 70, 100]
            }
        },
    }
}








