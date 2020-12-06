DATA_MATRIX = "Data\\DataMatrix_excel.xlsx" # Your file

FILE_TYPE = "excel" # File format progenesis, excel, csv or tsv. Check demo files if you are not sure which one to use.
USE_NORMALIZED = True  # If using progenesis file, otherwise it is not considered
N_SPLITS = 3 # The number of Monte-Carlo splits to generate for each design

EXPERIMENT_DESIGNS={
    "All Med vs N-A":{ # Don't use "_" in the exp name.
        "classes": {
            "North-American":["Alimed_A", "Med_A"],
            "Mediteranean": ["Alimed_B", "Med_B", "Med_C"]
        },
        "TestSize": 0.2,
        "positive_class": "Mediteranean"
    },
}
    
