DATA_MATRIX = "Data\\DataMatrix.csv"
METADATA = "Data\\sample_metadata_clean.xlsx"

EXPERIMENT_DESIGNS={
    "All Med-Na":{
        "classes": {
            "North-American":["Alimed_A", "Med_A"],
            "Mediteranean": ["Alimed_B", "Med_B", "Med_C"]
        },
        "TestSize": 0.2,
    },
    "Study Med":{
        "classes": {
            "North-American":["Med_A"],
            "Mediteranean": ["Med_B", "Med_C"]
        },
        "TestSize": 0.2,
    }
}
    
