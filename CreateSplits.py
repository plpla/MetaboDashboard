import numpy as np
import pandas as pd
import pickle as pkl
import glob, os
from multiprocessing import Pool
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix

from LearnConfig import *
from ExperimentDesign import *
from Utils import *

if __name__ == "__main__":
    # Read data matrix
    if FILE_TYPE == "progenesis":
        compounds_info, normalized, raw, labels, sample_names = read_Progenesis_compounds_table(DATA_MATRIX)
        if USE_NORMALIZED:
            df = normalized
        else:
            df = raw
    
    elif FILE_TYPE == "excel":
        compounds_info, df, labels, sample_names = read_excel_file(DATA_MATRIX)
    elif FILE_TYPE == "csv":
        compounds_info, df, labels, sample_names = read_text_file(DATA_MATRIX, sep=",")
    elif FILE_TYPE == "tsv":
        compounds_info, df, labels, sample_names = read_text_file(DATA_MATRIX, sep="\t")
    else:
        raise IOError("File type is unknown")

    # Prepare a bunch of splits.
    generate_splits(df, labels, compounds_info)