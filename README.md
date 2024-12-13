# capstone


## Runbook
1) ensure that data is in a `data/` directory in the root repo. Data in this folder will be ignore by git (per the .gitignore).

## Description of Files
- `preprocessing.py`: contains code to clean up reports data in Pandas dataframe and group them per patient ID
- `data_cleaner.py`: full data processing / data cleaning pipeline
- `naive_bayes.ipynb`: experimental work with a Naive Bayes classifier head