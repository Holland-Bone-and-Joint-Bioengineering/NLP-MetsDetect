import pandas as pd
import re
import html
# import spacy
# import torch
from tqdm import tqdm
from datetime import timedelta

class DataCleaner:
    def __init__(self, text_file, label_file):
        # Enable GPU usage in spaCy
        # if torch.cuda.is_available():
        #     print("using GPU"
        #     spacy.require_gpu()
        # self.nlp = spacy.load("en_core_web_sm")
        self.text_df = pd.read_csv(text_file)
        self.label_df = pd.read_csv(label_file)
        
    def load_data(
        self, 
        text_file='data/rad_report.csv', 
        label_file='data/labeled.csv'
    ):
        self.text_df = pd.read_csv(text_file)
        self.label_df = pd.read_csv(label_file)

    def clean_data(self, df):
        """
        Cleans the 'ReportText' column in the provided DataFrame by applying
        preprocessing on each text entry.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame with a 'ReportText' column to be cleaned.
        
        Returns:
        pd.DataFrame: The DataFrame with the cleaned 'ReportText' column.
        """
        if 'ReportText' in df.columns:
            tqdm.pandas(desc="Cleaning ReportText")
            df['ReportText'] = df['ReportText'].progress_apply(self.preprocess_single)
        else:
            raise ValueError("The DataFrame does not contain a 'ReportText' column.")
        
        return df

    def preprocess_single(self, text: str):
        """
        Preprocess a single text string.

        Inputs:
        - text : string, the body of an article

        Returns:
        - modified_text : string, the modified comment
        """
        escape_characters = {
            "\n": " ",
            "\t": " ",
            "\r": " ",
            "\b": " ",
            "\f": " ",
            "\v": " ",
            "\\": " ",
            "\a": " ",
            "\0": " ",
        }

        # Replace escape characters with spaces
        translation_table = str.maketrans(escape_characters)
        filtered_text = text.translate(translation_table)

        # Remove multiple spaces and trim leading/trailing spaces
        filtered_text = " ".join(filtered_text.split())

        # Convert text to lowercase
        filtered_text = filtered_text.lower()

        # Tokenize, remove stop words, punctuation, digits, and lemmatize using GPU
        return filtered_text

    def preprocess_reports(self):
        """
        Clean report data and group together reports for each patient.
        """
        print(self.label_df.columns)
        self.label_df.columns = self.label_df.columns.str.strip()
        text_df = self.text_df[['MRN', 'ServDescription', 'ReportDate', 'ReportText', 'studyinstanceuid']].dropna()
        label_df = self.label_df[['pt_shsc_id', 'imaging_date', 'study_uid', 'image_ct___1', 'image_ct___2']].dropna()
        

        # clean up data
        #label_df = label_df[~((label_df['image_ct___1'] == 0.0) & 
                          #  (label_df['image_ct___2'] == 0.0) & 
                           # (label_df['image_ct___3'] == 0.0))]
        #label_df.reset_index(drop=True, inplace=True)

        text_df['ReportDate'] = pd.to_datetime(text_df['ReportDate']).dt.date
        label_df['imaging_date'] = pd.to_datetime(label_df['imaging_date'], errors='coerce').dt.date

        # Filter `text_df` based on unique patient IDs in `label_df`
        unique_patient_ids = label_df['pt_shsc_id'].unique()
        text_df = text_df[text_df['MRN'].isin(unique_patient_ids)]

        # Drop rows with invalid or NaT dates in `label_df`
        label_df = label_df.dropna(subset=['imaging_date'])

        # Sort both DataFrames by patient ID and date columns
        text_df = text_df.sort_values(by=['MRN', 'ReportDate'])
        label_df = label_df.sort_values(by=['pt_shsc_id', 'imaging_date'])

        # Group reports for each patient by assessment periods
        grouped_results = []
        for patient_id in label_df['pt_shsc_id'].unique():
            # Filter reports and assessments for the current patient
            patient_reports = text_df[text_df['MRN'] == patient_id]
            patient_assessments = label_df[label_df['pt_shsc_id'] == patient_id].sort_values(by='imaging_date')
            
            # Initialize previous assessment date to a very early date --> to be able to chain together all prior reports for a person
            previous_assessment_date = previous_assessment_date = patient_reports['ReportDate'].min() - timedelta(days=365)
            print(patient_assessments.columns)

            # Iterate over each assessment date for this patient
            for _, assessment_row in patient_assessments.iterrows():
                current_assessment_date = assessment_row['imaging_date']
                #previous_assessment_date = current_assessment_date - timedelta(days=14)
                score1 = assessment_row['image_ct___1']
                score2 = assessment_row['image_ct___2']
                study_uid = assessment_row['study_uid']
                
                # Filter reports within the specified date range
                reports_in_range = patient_reports[
                    (patient_reports['ReportDate'] >= previous_assessment_date) &
                    (patient_reports['ReportDate'] <= current_assessment_date)
                ]
                #If no reports in specified date range, fall back on using study id to get report 
                if reports_in_range.empty:
                    reports_in_range = patient_reports[
                        patient_reports['studyinstanceuid'] == study_uid
                    ]

                print(f"Reports in range: {reports_in_range['ReportDate'].tolist()}")

                
                # Concatenate all relevant reports' text for the range
                concatenated_reports = " ".join(reports_in_range['ServDescription'] + " " + reports_in_range['ReportText']) if not reports_in_range.empty else ""

                grouped_results.append({
                    'patient_id': patient_id,
                    'assessment_date': current_assessment_date,
                    'study_uid': study_uid,
                    'reports': concatenated_reports,
                    'image_ct___1': score1,
                    'image_ct___2': score2,
                })
                
                # Update the previous assessment date
                previous_assessment_date = current_assessment_date
        print(f"Number of grouped results: {len(grouped_results)}")

        grouped_reports = pd.DataFrame(grouped_results)
        # remove any rows with empty strings
        dropped_rows = grouped_reports[grouped_reports['reports'] == ""]
        # Save them to a CSV for inspection
        dropped_rows.to_csv("data/dropped_reports.csv", index=False)
        grouped_reports = grouped_reports[grouped_reports['reports'] != ""]
        grouped_reports.reset_index(drop=True, inplace=True)
        grouped_reports.to_csv('data/grouped_test.csv')   #Output finalized csv 

        return grouped_reports



if __name__ == "__main__":
    data_cleaner = DataCleaner(
        text_file='data/rad_report.csv',
        label_file='data/labeled.csv'
    )
    data_cleaner.load_data()
    data_cleaner.text_df = data_cleaner.clean_data(data_cleaner.text_df)
    data_cleaner.preprocess_reports()

