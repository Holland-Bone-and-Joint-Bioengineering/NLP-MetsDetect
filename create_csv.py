import pandas as pd
import psycopg2
import numpy as np
from psycopg2 import sql
from psycopg2.extras import execute_batch

def add_radreports(DB_PARAMS):
    '''
     Load radiology reports for studies that also exist in the biomarkers table.

    Args:
        DB_PARAMS (dict): A dictionary of PostgreSQL database connection parameters.

    Returns:
        pd.DataFrame: A DataFrame containing radiology report text and metadata,
                      sorted by MRN and report date.
    '''
    conn = psycopg2.connect(**DB_PARAMS)
    print(f"connected to clinical")

    #Seelct radiology reports with biomarker data
    query = """
    SELECT r.*
    FROM oacis_radiologyreport r
    JOIN studies s ON r.studyinstanceuid = s.studyinstanceuid
    JOIN biomarkers_data_final b ON s.studyinstanceuid = b.study_uid
    WHERE r.studyinstanceuid IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    print(f"loaded oacis_radiologyreport")

    # sort reports by mrn and date
    df = df.sort_values(by=['mrn', 'reportdate'])

    return df

def create_labeled_data(DB_PARAMS):
    '''
    Create a labeled dataset combining radiology report text with vertebra-level metastasis and fracture labels.

    Args:
        DB_PARAMS (dict): A dictionary of PostgreSQL database connection parameters.

    Returns:
        pd.DataFrame: A DataFrame including patient/study info, report text, and labels for mets/fractures:
                      - 'image_ct___1' (fracture flag)
                      - 'image_ct___2' (metastasis flag)
    '''
    conn = psycopg2.connect(**DB_PARAMS)
    print(f"connected to clinical")

    #Query ground truth data 
    query = """
    SELECT 
    b.patient_uid,
    b.study_date,
    smr.study_uid,
    r.reporttext,
    smr.l1,
    smr.l2,
    smr.l3,
    smr.l4,
    smr.l5,
    smr.l1_frac,
    smr.l2_frac,
    smr.l3_frac,
    smr.l4_frac,
    smr.l5_frac
    FROM oacis_radiologyreport r
    JOIN studies s ON r.studyinstanceuid = s.studyinstanceuid
    JOIN biomarkers_data_final b ON s.studyinstanceuid = b.study_uid
    LEFT JOIN scan_manual_review smr ON b.study_uid = smr.study_uid
    WHERE  
    smr.l1 IS NOT NULL AND 
    smr.l2 IS NOT NULL AND 
    smr.l3 IS NOT NULL AND 
    smr.l4 IS NOT NULL AND 
    smr.l5 IS NOT null
    """

    '''
    QUERY FOR ALL BIOMARKER LABELS TO CREATE WHOLE SET:
    
    query = """
    SELECT 
    b.patient_uid,
    b.study_date,
    b.study_uid,
    r.reporttext,
    smr.l1,
    smr.l2,
    smr.l3,
    smr.l4,
    smr.l5,
    smr.l1_frac,
    smr.l2_frac,
    smr.l3_frac,
    smr.l4_frac,
    smr.l5_frac
    FROM biomarkers_data_final b
    LEFT JOIN studies s ON b.study_uid = s.studyinstanceuid
    LEFT JOIN oacis_radiologyreport r ON r.studyinstanceuid = s.studyinstanceuid
    LEFT JOIN scan_manual_review smr ON b.study_uid = smr.study_uid
    WHERE r.reporttext is not null 
    """
    '''

    df = pd.read_sql_query(query, conn)
    print(f"loaded labeled data ")
    cols_to_check_mets = ['l1', 'l2', 'l3', 'l4', 'l5']

    cols_to_check = ['l1_frac', 'l2_frac', 'l3_frac', 'l4_frac', 'l5_frac']

    def compute_flag(row, cols):
        #Function to return 1 for image__ct1 or 2 if any of the vertebrae specific values are 1 and 0 if they are all 0
        if row[cols].isna().all():
            return -1
        elif (row[cols] == 1).any():
            return 1
        else:
            return 0

    #Create fracture labels
    df['image_ct___1'] = df.apply(lambda row: compute_flag(row, cols_to_check), axis=1)

    #Create mets labels
    df['image_ct___2'] = df.apply(lambda row: compute_flag(row, cols_to_check_mets), axis=1)

    #Define and drop columns that are not necessary 
    cols_to_drop = [
        "l1_frac",
        "l2_frac",
        "l3_frac",
        "l4_frac",
        "l5_frac"
    ]

    df = df.drop(columns = cols_to_drop)

    return df

# Example usage
if __name__ == "__main__":

    DB_PARAMS = {
        "dbname": "EMR_ProstateStudy_Combined",
        "user": "",
        "password": "",
        "host": "", 
        "port": "5432", 
    }

    #Create rad report dataframe
    df_rad_report = add_radreports(DB_PARAMS)

    rename_map = {
        'mrn': 'MRN',
        'oacis_pid': 'OACIS_PID',
        'oacis_sid': 'OACIS_SID',
        'oacis_tid': 'OACIS_TID',
        'servacro': 'ServAcro',
        'servdescription': 'ServDescription',
        'reportdate': 'ReportDate',
        'reporttext': 'ReportText'
    }

    #Rename columns for use in data_cleaner.py
    df_rad_report.rename(columns=rename_map, inplace=True)
    df_rad_report.to_csv('rad_report.csv')

    #Create labeled data dataframe 
    df_labeled = create_labeled_data(DB_PARAMS)

    rename_map_labeled = {
        'patient_uid': 'pt_shsc_id',
        'study_date': 'imaging_date',
        'reporttext': 'reports'
    }

    #rename columns 
    df_labeled.rename(columns=rename_map_labeled, inplace=True)
    df_labeled.to_csv('labeled.csv')


