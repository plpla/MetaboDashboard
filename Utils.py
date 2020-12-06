import pandas as pd
import numpy as np
import pickle as pkl
import random, os
from LearnConfig import *
from ExperimentDesign import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def read_excel_file(filename):
    datatable = pd.read_excel(filename, header=1, index_col= 0)

    targets_df = pd.read_excel(filename, index_col= 0, nrows=1, header=None)
    labels_array = np.array(targets_df.iloc[0].tolist())
    labels = labels_array[labels_array != 'nan']
    num_columns_before_first_sample = targets_df.shape[1] - len(labels)

    sample_names = datatable.iloc[:, num_columns_before_first_sample:].columns
    compounds_info = datatable.iloc[:, :num_columns_before_first_sample]
    df = datatable.iloc[:num_columns_before_first_sample:]

    # Before return make sure df and labels have the same size. 
    if datatable.shape[1] != len(labels_array):
        raise ValueError("Df and targets don't have the same length. Make sure your file is formated properly.")

    return compounds_info, df, labels, sample_names


def read_text_file(filename, sep=","):
    datatable = pd.read_table(filename, header=1, index_col= 0, sep=sep)
    targets_df = pd.read_table(filename, index_col= 0, nrows=1, header=None, sep=sep)

    labels_array = np.array(targets_df.iloc[0].tolist())
    labels = labels_array[labels_array != 'nan']
    num_columns_before_first_sample = targets_df.shape[1] - len(labels)

    sample_names = datatable.iloc[:, num_columns_before_first_sample:].columns
    compounds_info = datatable.iloc[:, :num_columns_before_first_sample]
    df = datatable.iloc[:num_columns_before_first_sample:]

    # Before return make sure df and labels have the same size. 
    if datatable.shape[1] != len(labels_array):
        raise ValueError("Df and targets don't have the same length. Make sure your file is formated properly.")

    return compounds_info, df, labels, sample_names


def read_Progenesis_compounds_table(fileName):
    datatable = pd.read_csv(fileName, header=2, index_col=0)
    header = pd.read_csv(fileName, nrows=1, index_col=0)
    start_normalized = header.columns.tolist().index("Normalised abundance")
    start_raw = header.columns.tolist().index("Raw abundance")
    
    labels_array = np.array(header.iloc[0].tolist())
    possible_labels = labels_array[labels_array != 'nan']
    possible_labels = possible_labels[0:int(len(possible_labels) / 2)]
    
    sample_names = datatable.iloc[:, start_normalized:start_raw].columns
    
    labels = [""] * len(sample_names)
    start_label = possible_labels[0]
    labels_array = labels_array.tolist()
    for next_labels in possible_labels[1:]:
        index_s = labels_array.index(start_label) - start_normalized
        index_e = labels_array.index(next_labels) - start_normalized
        labels[index_s : index_e] = [start_label] * (index_e - index_s)
        start_label = next_labels
    labels[index_e:] = [start_label] * (len(labels) - index_e)

    datatable_compoundsInfo = datatable.iloc[:,0:start_normalized]
    datatable_normalized = datatable.iloc[:,start_normalized:start_raw]
    datatable_raw = datatable.iloc[:,start_raw:]
    datatable_raw.columns = [i.rstrip(".1") for i in datatable_raw.columns] #Fix the columns names when normalized and raw are present

    datatable_normalized = datatable_normalized.T
    datatable_raw = datatable_raw.T
    datatable_compoundsInfo = datatable_compoundsInfo.T
    datatable_normalized.rename(columns={"Compound": "Sample"})
    datatable_raw.rename(columns={"Compound": "Sample"})
    return datatable_compoundsInfo, datatable_normalized, datatable_raw, labels, sample_names

def filter_sample_based_on_labels(data, labels, labels_to_keep):
    labels_filter = np.array([i in labels_to_keep for i in labels])
    d = data.iloc[labels_filter]
    l = np.array(labels)[labels_filter]
    return d, l

def generate_splits(df, labels, compounds_info):
    # TODO: Make this lighter
    for design_name in EXPERIMENT_DESIGNS:
        label = np.array(labels)
        classes = list(EXPERIMENT_DESIGNS[design_name]["classes"].keys())
        group_to_class = get_group_to_class(EXPERIMENT_DESIGNS[design_name]["classes"])
        groups_to_keep = [item for sublist in EXPERIMENT_DESIGNS[design_name]["classes"].values() for item in sublist]

        design_df, labels_design = filter_sample_based_on_labels(df, labels, groups_to_keep)
        labels_design = [group_to_class[i] for i in labels_design]
        for split_number in range(LEARN_CONFIG["Nsplit"]):
            X_train, X_test, y_train, y_test = train_test_split(design_df, \
                labels_design, test_size=EXPERIMENT_DESIGNS[design_name]["TestSize"])
            # Write split to files
            with open(os.path.join("Splits", "{}_{}".format(design_name, split_number)), "wb") as fo:
                pkl.dump(X_train, fo)
                pkl.dump(y_train, fo)
                pkl.dump(X_test, fo)
                pkl.dump(y_test, fo)
                pkl.dump(compounds_info)

def run_learning_job(job_config):
    # (filename, algo_config_name, algo_function, grid)
    filename = job_config[0]
    design_name = os.path.split(filename)[1].split("_")[0]
    split_number = os.path.split(filename)[1].split("_")[1]
    algo_config_name = job_config[1]
    learning_function = job_config[2]
    param_grid = job_config[3]
    print("Processing {} using design {}".format(filename, design_name))

    # Load file
    with open(filename, "rb") as fi:
        train_df = pkl.load(fi)
        train_targets = pkl.load(fi)
        test_df = pkl.load(fi)
        test_targets = pkl.load(fi)

    # Run grid search.
    gc = GridSearchCV(learning_function(), param_grid, cv=LEARN_CONFIG["CV_folds"])
    gc.fit(train_df, train_targets)
    # Predict on train.
    train_predict = gc.predict(train_df)

    # Predict on test.
    test_predict = gc.predict(test_df)

    # Save to file.
    with open(os.path.join("Results", 
        "{}_{}_{}.pkl".format(design_name, split_number, algo_config_name)), "wb") as fo:
        pkl.dump(gc, fo)
        pkl.dump(train_predict, fo)
        pkl.dump(test_predict, fo)


def get_group_to_class(classes):
    group_to_class = {}
    for class_name in classes:
        for subgroup in classes[class_name]:
            group_to_class[subgroup] = class_name
    return group_to_class