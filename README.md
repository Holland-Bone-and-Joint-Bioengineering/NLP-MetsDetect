# capstone


## How to use Llama pipeline
1) Run `data\create_csv.py` to generate rad_reports and labeled data csvs
2) Run `data_cleaner.py` using rad reports and labeled data csvs to create final grouped csv
3) Pass the final grouped csv into `fewshot_llama.py` 
4) Use resultant csv

## Description of Files
- `create_csv.py`: contains code to create dataframes with reports and labels
- `data_cleaner.py`: full data processing / data cleaning pipeline
- `fewshot_llama.py`: NLP processing using LLama and fewshot learning to detect metastases and fracture based on radiology reports 

## Requirements
1) Run conda env create -f environment.yml
2) Download llama model Llama-3.1-8B-Instruct 
3) Store llama model in .cache directory 
4) Get PostgreSQL database access 