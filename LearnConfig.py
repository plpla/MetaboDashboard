import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier

LEARN_CONFIG={
    "FileType": "progenesis", # progenesis, excel, csv or tsv
    "UseNormalized": True, # If using progenesis file. Otherwise it is not considered
    "Nsplit": 3,
    "CV_folds": 3,
    "Algos":{
        "DecisionTree":{
            "function": DecisionTreeClassifier,
            "ParamGrid": {
                "max_depth": [1, 3, 5, 10],
                "min_samples_split": [2, 5, 10]
            }
        },
        "RandomForest":{
            "function": RandomForestClassifier,
            "ParamGrid": {
                "n_estimators": [30, 100]
            }
        },
    }
}








