from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, roc_auc_score, recall_score

NUMBER_FEATURE_TO_KEEP_FOR_PCA = 40

def true_positive_rate(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)[1][1]

def false_positive_rate(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)[0][1]

def true_negative_rate(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)[0][0]

def false_negative_rate(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)[1][0]

STATISTICS={"Accuracy": accuracy_score,
            "ROC AUC Score": roc_auc_score,
            #"f1-score": f1_score,
            #"Precision": precision_score,
            #"Sensitivity": recall_score, #Positive predicted correctly (TP/TP+FN)
            "True positive": true_positive_rate, #Positive prediction that are positive (TP/TP+FP)
            "False positive": false_positive_rate,
            "True negative": true_negative_rate,
            "False negative": false_negative_rate}

