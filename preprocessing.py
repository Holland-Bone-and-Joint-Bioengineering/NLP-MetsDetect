import pandas as pd


def preprocess_data(report_file, labels_file):
    # Load data
    text_df = pd.read_csv(report_file)
    label_df = pd.read_csv(labels_file)
    
    # Select specific columns
    text_df = text_df[['MRN', 'ServDescription', 'ReportDate', 'ReportText']].dropna()
    label_df = label_df[['pt_shsc_id', 'imaging_date', 'image_ct___1', 'image_ct___2', 'image_ct___3']].dropna()

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
    return grouped_reports


if __name__=="__main__":
    report_file = 'data/[DEIDENTIFIED]OACIS_RadiologyReport_20241024.csv'
    labels_file = 'data/Osteosarc Rad Report Data Oct 7th.csv'

    grouped_reports = preprocess_data(report_file, labels_file)
    print(grouped_reports)