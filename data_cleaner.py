import pandas as pd
import re
import html
# import spacy
# import torch
from tqdm import tqdm

class DataCleaner:
    def __init__(self):
        # Enable GPU usage in spaCy
        # if torch.cuda.is_available():
        #     print("using GPU"
        #     spacy.require_gpu()
        # self.nlp = spacy.load("en_core_web_sm")
        pass

    def load_data(
        self, 
        text_file='data/[DEIDENTIFIED]OACIS_RadiologyReport_20241024.csv', 
        label_file='data/Osteosarc Rad Report Data Oct 7th.csv'
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
        # Select specific columns
        text_df = self.text_df[['MRN', 'ServDescription', 'ReportDate', 'ReportText']].dropna()
        label_df = self.label_df[['pt_shsc_id', 'imaging_date', 'image_ct___1', 'image_ct___2', 'image_ct___3']].dropna()

        # Remove rows where all scores are zero
        label_df = label_df[~((label_df['image_ct___1'] == 0.0) & 
                            (label_df['image_ct___2'] == 0.0) & 
                            (label_df['image_ct___3'] == 0.0))]
        label_df.reset_index(drop=True, inplace=True)

        # Filter `text_df` based on unique patient IDs in `label_df`
        unique_patient_ids = label_df['pt_shsc_id'].unique()
        text_df = text_df[text_df['MRN'].isin(unique_patient_ids)]

        # Convert date columns to datetime
        text_df['ReportDate'] = pd.to_datetime(text_df['ReportDate']).dt.date
        label_df['imaging_date'] = pd.to_datetime(label_df['imaging_date'], format='%m/%d/%Y', errors='coerce').dt.date

        # Drop rows with invalid or NaT dates in `label_df`
        label_df = label_df.dropna(subset=['imaging_date'])

        # Sort both DataFrames by patient ID and date columns
        text_df = text_df.sort_values(by=['MRN', 'ReportDate'])
        label_df = label_df.sort_values(by=['pt_shsc_id', 'imaging_date'])

        # Apply data cleaning to text
        text_df['ReportText'] = text_df['ReportText'].apply(self.preprocess_single)

        # Initialize list for grouped results
        grouped_results = []

        # Group reports for each patient by assessment periods
        for patient_id in label_df['pt_shsc_id'].unique():
            # Filter reports and assessments for the current patient
            patient_reports = text_df[text_df['MRN'] == patient_id]
            patient_assessments = label_df[label_df['pt_shsc_id'] == patient_id].sort_values(by='imaging_date')
            
            # Initialize previous assessment date to a very early date
            previous_assessment_date = pd.Timestamp.min.date()
            
            # Iterate over each assessment date for this patient
            for _, assessment_row in patient_assessments.iterrows():
                current_assessment_date = assessment_row['imaging_date']
                '''
                from datetime import timedelta
                previous_assessment_date = current_assessment_date - timedelta(days=14)'''

                # Get scores from assessment row
                score1 = assessment_row['image_ct___1']
                score2 = assessment_row['image_ct___2']
                score3 = assessment_row['image_ct___3']
                
                # Filter reports within the specified date range
                reports_in_range = patient_reports[
                    (patient_reports['ReportDate'] > previous_assessment_date) &
                    (patient_reports['ReportDate'] <= current_assessment_date)
                ]
                
                # Concatenate all relevant reports' text for the range
                concatenated_reports = " ".join(reports_in_range['ServDescription'] + " " + reports_in_range['ReportText']) if not reports_in_range.empty else ""
                
                # Append the results as a dictionary to the list
                grouped_results.append({
                    'patient_id': patient_id,
                    'assessment_date': current_assessment_date,
                    'reports': concatenated_reports,
                    'image_ct___1': score1,
                    'image_ct___2': score2,
                    'image_ct___3': score3
                })
                
                # Update the previous assessment date
                previous_assessment_date = current_assessment_date


        # Convert the results list to a DataFrame
        grouped_reports = pd.DataFrame(grouped_results)

        # Remove rows where the 'reports' column is an empty string
        grouped_reports = grouped_reports[grouped_reports['reports'] != ""]
        grouped_reports.reset_index(drop=True, inplace=True)

        # Display the grouped reports DataFrame
        print(grouped_reports)



if __name__ == "__main__":
    data_cleaner = DataCleaner()
    data_cleaner.load_data()
    data_cleaner.preprocess_reports()

