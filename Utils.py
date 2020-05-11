import pandas as pd
import numpy as np
import pickle as pkl
import random, os
from LearnConfig import *
from ExperimentDesign import *

from sklearn.model_selection import GridSearchCV


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
    
    labels_dict = {sample_names[i] : j for i,j in enumerate(labels)}

    datatable_compoundsInfo = datatable.iloc[:,0:start_normalized]
    datatable_normalized = datatable.iloc[:,start_normalized:start_raw]
    datatable_raw = datatable.iloc[:,start_raw:]
    datatable_raw.columns = [i.rstrip(".1") for i in datatable_raw.columns] #Fix the columns names

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

def generate_splits(df, labels):
    # TODO: Use index instead of full table. Maybe put everything in a single file?
    for design_name in EXPERIMENT_DESIGNS:
        label = np.array(labels)
        classes = list(EXPERIMENT_DESIGNS[design_name]["classes"].keys())
        group_to_class = get_group_to_class(EXPERIMENT_DESIGNS[design_name]["classes"])
        groups_to_keep = [item for sublist in EXPERIMENT_DESIGNS[design_name]["classes"].values() for item in sublist]
        design_df, labels_design = filter_sample_based_on_labels(df, labels, groups_to_keep)
        study_unique_patients_id = set(design_df["subject"])
        for split_number in range(LEARN_CONFIG["Nsplit"]):
            test_subjects_id = random.sample(study_unique_patients_id, 
                int(EXPERIMENT_DESIGNS[design_name]["TestSize"]*len(study_unique_patients_id)))
            train_subjects_id = [i for i in study_unique_patients_id if i not in test_subjects_id]

            # Create train-test mask and create x_train, y_train, x_test, y_test
            train_mask = np.array([i in train_subjects_id for i in design_df["subject"]])
            test_mask = np.array([i in test_subjects_id for i in design_df["subject"]])
            train_df = design_df[train_mask]
            train_targets = [group_to_class[i] for i in labels_design[train_mask]]
            test_df = design_df[test_mask]
            test_targets = [group_to_class[i] for i in labels_design[test_mask]]

            # Write split to files
            with open(os.path.join("Splits", "{}_{}".format(design_name, split_number)), "wb") as fo:
                pkl.dump(train_df, fo)
                pkl.dump(train_targets, fo)
                pkl.dump(test_df, fo)
                pkl.dump(test_targets, fo)
                pkl.dump(train_subjects_id, fo)
                pkl.dump(test_subjects_id, fo)

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
    train_df = train_df.drop(["subject"], axis=1) # We don't want to learn something using the Subject column.
    test_df = test_df.drop(["subject"], axis=1)
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