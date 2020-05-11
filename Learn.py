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
    compounds, normalized, raw, labels, sample_names = read_Progenesis_compounds_table(DATA_MATRIX)

    # Read metadata
    metadata = pd.read_excel(METADATA, index_col=0)

    # Merge data matrix and metadata
    if LEARN_CONFIG["UseNormalized"]:
        df = normalized.merge(right=metadata, how="left", right_index=True, left_index=True)
    else:
        df = raw.merge(right=metadata, how="left", right_index=True,left_index=True)

    # Prepare a bunch of splits.
    generate_splits(df, labels)

    # Create a list of jobs. Jobs are a set with the following structure:
    # (filename, algo_config_name, algo_function, grid)
    learning_job_list = []
    for filename in glob.glob(os.path.join("Splits", "*")):
        for algo_config in LEARN_CONFIG["Algos"]:
            learning_job_list.append((filename, 
                algo_config,
                LEARN_CONFIG["Algos"][algo_config]["function"],
                LEARN_CONFIG["Algos"][algo_config]["ParamGrid"]))

    pool = Pool(4)
    pool.map(run_learning_job, learning_job_list[0:3])
    #run_learning_job(learning_job_list[0])
    



