import pandas as pd
import re
import html
import spacy
import torch
from tqdm import tqdm

class DataCleaner:
    def __init__(self):
        # Enable GPU usage in spaCy
        # if torch.cuda.is_available():
        #     print("using GPU")
        #     spacy.require_gpu()
        self.nlp = spacy.load("en_core_web_sm")

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
