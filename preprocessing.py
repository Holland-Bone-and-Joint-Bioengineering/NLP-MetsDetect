import pandas as pd

# Load data
text_df = pd.read_csv('data/[DEIDENTIFIED]OACIS_RadiologyReport_20241024.csv')
label_df = pd.read_csv('data/Osteosarc Rad Report Data Oct 7th.csv')

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
print(grouped_reports)




'''
import re

def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        # Replace non-alphabetic characters with space
        cleaned_text = re.sub(r'[^a-zA-Z ]+', ' ', text)
        # Remove extra spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Remove words with 2 or fewer characters, except for 'CT'
        filtered_words = [word for word in cleaned_text.split() if len(word) > 2 or word == 'CT']
        processed_texts.append(' '.join(filtered_words))
        
    return processed_texts


# Example usage
phrases = [
    'CT chest C+ -C','S52 MRI CTL Spine No Contrast -S','Head CT 1P C- -NU','CT chest C- -C','Head CT 1P C-C+ -NU','Bone Scan WB+Flow+Spect -NM','CTPA C-C+ -C','H26C  MRI Brain with contrast -NU','Bone Scan Whole Body+Spect+No Flow -NM','C-Chest C+','Neck Soft Tissue CT 2P C+ -NU','BP8C MRI Pelvis with Contrast -B','H12C  MRI Brain with contrast -NU','M6  MRI Pelvis No Contrast -M','Cervical Spine CT C- -S','CT Head 1P C-C+ & Neck 2P C+ -NU','CTA  Head and Neck -NU','H6C  MRI Brain with contrast -NU','Thoracic Spine CT C- -S','Multiphasic liver CT C-C+ -B','S54C MRI CTL Spine With Contrast -S','S61 MRI Spine ROI - Pre Treatment -S','C-Chest C-','CT chest C+','Outside Images -X','CT Perfusion C+& Head C-C+& CTA Carotid with 3D -NU','CT Urogram -B','H36 MRI Brain no Contrast -NU','S42 MRI Lumbar Spine No Contrast -S','H19C  MRI Neck with contrast -NU','NU-HeadCT 1 wow','C-CTPA C-C+','CT: Urogram -B','Chest biopsy, CT guided -A','Femur CT: Bil C- -M','MR Minor Assessment -X','CT chest C-C+ -C','PET RESEARCH -NM','CTA AIF -A','S53C MRI CTL Spine With Contrast -S','TR-Head + C.Spine C- -NU','Femur CT : LT C- -M','M25 MRI Pelvis No Contrast -M','NU-Head CT 1 WO','BA3C  MRI Abdomen  With Contrast -B','CT chest C-','H18C  MRI Brain with contrast -NU','Head CT 1 P C+ -NU','Orbit 2P + Head CT C+ -NU','S40 MRI Cervical Spine No Contrast -S','2nd opinion on CT study, per exam -X','BP7C MRI Pelvis with Contrast -B','NU-3D Brain WC','NU-NeckST CT2wc','PET Single Pulmonary Nodule -NM','BA16C MRI Abdomen with Contrast -B','CT chest C-C+','CTA Gated Aorta -H','CTA Head -NU','H17C  MRI Brain with contrast -NU','H2  MRI Brain No contrast -NU','H9  MRI Brain No contrast -NU','Heart Persantine Stress+Rest -NM','Neck Soft Tissue CT 2P C- -NU','PET Non Small Cell Lung Cancer -NM','S-Cerv CT wo c','2nd opinion on MR study, per exam -X','BA14C MRI Abdomen with Contrast -B','BP3C  MRI Pelvis and Abdomen  With Contrast -B','CT Enterography C+ -B','CTV Head + Head C-C+ & 3D -NU','Day Two of Two-Day Heart Scan -NM','Day one of two day heart scan -NM','Facial Bones CT 2P C+ -NU','Femur CT : RT C- -M','H10 MRI Cardiac no Contrast -H','H11C  MRI Brain with contrast -NU','H14  MRI Brain No contrast -NU','H33C  MRI Brain and Neck with contrast -NU','H9C  MRI Brain with contrast -NU','Head1P & Neck 2P C- -NU','M23 MRI Pelvis and CTL Spine No Contrast -M','M9  MRI Left Hip No Contrast -M','S41 MRI Thoracic Spine No Contrast -S','A-3D VascExt WC','B-3dABDO MR W C','BA5  MRI Abdomen  No Contrast -B','BP15 MRI Pelvis No Contrast -B','CT KUB Low Dose C- -B','CTA chest C-C+ -A','CTA coronary or cardiac -H','Emerg - Head CT -NU','Enterography CT:C+ -B','Facial Bone 2P C- & 3D  CT -NU','H32C  MRI Brain with contrast -NU','Head CT 2 P C-C+ -NU','Heart Scan Stress+Rest NM -NM','M11  MRI Right Lower Extremity No Contrast -M','Petrous Bones CT 2P Bi C- -NU','S52 MRI CTL Spine No Contrast','S60 MRI CTL Spine No Contrast -S','Shoulder CT: Lt C- -M','Trauma CT series -X','B- CT UROGRAM','BA15C MRI Abdomen with Contrast -B','BA19C MRI Abdomen with Contrast -B','BA9  MRI Abdomen  with Contrast -B','BP100 MRI Pelvis Limited -B','BP12 MRI Pelvis no Contrast -B','BP7  MRI Pelvis No Contrast -B','BP9C MRI Pelvis with Contrast -B','CT Colonography -B','CT Colonography C- -B','CT pre osteoplasty -A','Chest biopsy, CT guided -C','Elbow CT : Lt C- -M','Femur CT Bilateral -M','Femur CT Bilateral C+ -M','H13C  MRI Brain with contrast -NU','H19  MRI Neck No contrast -NU','H7C MRI Cardiac with Contrast -H','H9C MRI Cardiac with Contrast -H','Head & Facial Bones 2P CT C- -NU','Head & FacialB 2P & CSpine CT C- -NU','Head CT 2P C- -NU','Knee Ct : Rt C- -M','M12C  MRI Left Lower Extremity With Contrast -M','M19  MRI Left Shoulder No Contrast -M','M24 MRI Chest (MSK) without Contrast -M','M5  MRI Right Knee No Contrast -M','M7 MRI Pelvis No Contrast -M','NU-3D Neck WC','NU-Brain wo MRI','NU-TR-HeadCSpwo','Outside Images','PET Residual Mass Lymphoma -NM','RFA liver -A','S-ThoracS CT WO','S43C MRI Lumbar Spine With Contrast -S','Shoulder CT: Rt C+ -M','Thoracic Spine CT C+ -S','Tibia/Fibula CT: Lt C- -M','Tibia/Fibula CT: Rt C- -M','3D Reconstruct: Face -NU','A-3D Aorta T wc','A-Chest C-','A9C MRI Pelvis Angiogram -A','Ankle CT: LT C- -M','B-3d Pel MR WC','B-3d Pel-EndC+','B-3d Pel-ProsC','B-Cysto CT wc','BA2C  MRI Abdomen  With Contrast -B','BA4C  MRI Abdomen  With Contrast -B','BP1   MRI Pelvis and Abdomen No Contrast','BP10C MRI Pelvis with Contrast -B','BP2C  MRI Pelvis and Abdomen  With Contrast -B','BP4  MRI Pelvis No Contrast -B','BP5C  MRI Pelvis and Abdomen  With Contrast -B','BP7  MRI Pelvis No Contrast','Bone Scan(Whole Body)Nuc Med -NM','Brain SPECTNo Flow -NM','C1C MRI Chest With Contrast -C','C3C MRI Chest With Contrast -C','CT Colonography C-C+ -B','CT Guid Abscess / Cyst Drainage:Pelvis -B','CT: Urogram','Cervical Spine CT -S','Cervical Spine CT C-','Chest CT -C','Emerg - Cervical Spine CT -S','Enterography CT:C+','FDG Melanoma -NM','Facial Bones CT 2P C- -NU','Femur CT Left C+ -M','Femur CT Right C+ -M','H-CCTA','H10 MRI Cardiac with Contrast -H','H10C  MRI Brain with contrast -NU','H11  MRI Brain No contrast -NU','H26C  MRI Brain with contrast','H3  MRI Brain No contrast -NU','H31  MRI Brain No contrast -NU','H34  MRI Brain and Neck No contrast -NU','H36  MRI Brain and Neck No contrast -NU','H6C  MRI Brain with contrast','H8C MRI Cardiac with Contrast -H','Head CT 1P C-C+','Head CT 2 P C+ -NU','Head CT 2P -NU','Heart Pharm, Stress Only NM -NM','Heart PharmStress RestGSpect -NM','Heart Rest Only NM -NM','Heart Scan& SR Gated Spect:NM -NM','Humerus CT: Lt C+ -M','Humerus CT: Lt C- -M','Knee Ct :  Bil C- -M','Lung Perfusion Scan -NM','M-CT BX LowE WO','M-FemurCT BIwoc','M-HumerusCT Lwo','M-MSK Pelvis WO','M-Pelvis WO +3D','M103 MRI Upper Extremity Bilateral -M','M10C  MRI Right Hip With Contrast -M','M11  MRI Left Lower Extremity No Contrast -M','M12C  MRI Right Lower Extremity With Contrast -M','M19  MRI Right Shoulder No Contrast -M','M7C  MRI Pelvis With Contrast','M8  MRI Bilateral Hips No Contrast -M','M9  MRI Right Hip No Contrast -M','Microwave ablation-liver -A','Minor assessment -X','N-BoneScan WB','NU-3DBra+Spe wc','NU-CTV Head +3D','NU-FaciB CT 2wc','NU-He+fac2+CSwo','NU-Head CT 2 WO','NU-Head CT 2 wc','NU-OrbitHCT2Pwc','NU-PerHeWO+CTA3','NU-SinusCT 2Pwo','Neck Soft Tissue CT 1P C+ -NU','Neck Soft Tissue CT 1P C- -NU','Orbit 2P + Head CT C- -NU','Orbit 2P + Head CT C-C+ -NU','PET Small Cell -NM','PETMOH -NM','Pelvis + Cystogram CT C+ -B','Pleural drain placement -A','RFA kidney -A','S-3D CerS WO MR','S-ThoracS CT wc','S100 MRI Spine Limited -S','S101 MRI Spine Intermediate -S','S205 MRI C-T-L Spine','S205 MRI C-T-L Spine -S','S47C MRI Cervical Spine With Contrast -S','S48C MRI Thoracic Spine With Contrast -S','S49C MRI Lumbar Spine With Contrast -S','S53 MRI CTL Spine No Contrast -S','Shoulder CT Right C- -M','Shoulder CT without contrast: BIL -M','Shoulder CT: Bil C+ -M','Sinuses Ct 2P C- -NU','Thoracic Spine CT -S','Tibia/Fibula CT: Lt C+ -M', 'S-CTL Spine woC', 'Post Vertebroplasty CT -S','Pelvis CT C+ -M','Pelvis CT C- (B) -B','Pelvis CT C+ (B) -B','Pelvis CT C-','Pelvis Ct C+ (B) -B','Pelvis CT C- -M','Pelvis Ct C+ (B) -B', 'CT Biopsy: Pelvis (bone) -M','CT Biopsy:Abdomen -B','CT Pelvis (MSK) & Biopsy -M','CT Spine & biopsy -S','CT Biopsy: Pelvis C- -B','CT Biopsy: Pelvis C-',
]

cleaned_phrases = preprocess_text(phrases)
#for phrase in cleaned_phrases:
    #print(phrase)

'''
