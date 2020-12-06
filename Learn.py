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
    # Create a list of jobs. Jobs are a set with the following structure:
    # (filename, algo_config_name, algo_function, grid)
    learning_job_list = []
    for filename in glob.glob(os.path.join("Splits", "*")):
        for algo_config in LEARN_CONFIG["Algos"]:
            learning_job_list.append((filename, 
                algo_config,
                LEARN_CONFIG["Algos"][algo_config]["function"],
                LEARN_CONFIG["Algos"][algo_config]["ParamGrid"]))

    pool = Pool(6)
    pool.map(run_learning_job, learning_job_list)
    #run_learning_job(learning_job_list[0])
    



