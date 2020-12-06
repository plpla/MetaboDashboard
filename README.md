# MetaboDashboard
Simplified machine learning for metabolomics.

## Installation ##
- Download or clone the git repository.

- Install Miniconda (see https://docs.conda.io/en/latest/miniconda.html)

- Create a virtual environment to use MetaboDashboard:
```
conda env create -f environment.yml
```

Make sure to activate the environment when using MetaboDashboard
```
conda activate metabodashboard
```

## Usage ##
MetaboDashboard separates the analysis in three independent steps: data preparation, machine learning model generation and models evaluation. Parameters are changed through configuration files.

### Data preparation ###
The objective here is to create train/test sets that can be then used to train machine learning models. To this end, you must specify the file containing your data and the experiment parameters in the `ExperimentDesign.py` file:
```
DATA_MATRIX = "Data\\DataMatrix.csv" # Your file

FILE_TYPE = "progenesis" # File format progenesis, excel, csv or tsv. Check demo files if you are not sure which one to use.
USE_NORMALIZED = True  # If using progenesis file, otherwise it is not considered
N_SPLITS = 3 # The number of Monte-Carlo splits to generate for each design

EXPERIMENT_DESIGNS={
    "All Med-Na":{
        "classes": {
            "North-American":["Alimed_A", "Med_A"],
            "Mediteranean": ["Alimed_B", "Med_B", "Med_C"]
        },
        "TestSize": 0.2,
        "positive_class": "Mediteranean"
    },
}
```

The `EXPERIMENT_DESIGNS` is a python `dict` containing a machine learning experiment definition. In this example, the experiment name is `All Med-Na`. It contains two classes that we want the models to distinguish: `North-American` and `Mediteranean`. Each of these classes are composed of multiple sublasses. In this case, `Alimed_A` and `Med_A` are similar samples (North-American diet) from two independant studies that we group together.

The `TestSize` parameter is the fraction of the full dataset used in the test set. The `positive_class` parameter is the name of the class we want to use for metrics computation like `AUC`.

Once the deisgn is completed, you can generate the splits using:
```
python CreateSplits.py
```

### Machine learning models ###
Now that we have generated some data splits, we want to use then to train machine learning models. The different machine learning algorithm that we want to use are specified in the `LearnConfig.py` file.

The file structure is pretty similar to the ExperimentDesign.py. Just make sure to import the algorithm you want to use in the file header.

Once completed, juste run:
```
python Learn.py
```

### Look at the results ###
Now that we have some results, you can start the dashboard app to compare the models and metrics.
```
python MetaboDashboard.py
```
