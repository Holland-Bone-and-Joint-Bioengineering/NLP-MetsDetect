#     report = """
# Bone Scan Whole Body+Extra Views+Flow TECHNETIUM MDP BONE SCAN WHOLE BODY:

# Preliminary flow study of the pelvis demonstrates asymmetry in the
# flow to the legs with more flow to the right leg compared to the
# left this is often due to altered weight bearing.

# Static whole body images demonstrate mild increased activity at
# T8-T9, the patellofemoral joints and the left first MTP joint in
# keeping with arthritis. The findings are stable when compared
# with bone scans as far back as 12/19/06.

# No scintigraphic evidence of bony metastasis is seen.

# [DEIDENTIFIED - DOCTOR'S INFO]
# """
#     frac, met = predict_label_lama(report)
#     actual_f, actual_m = 0, 0
#     print(f"Predicted - fracture: {frac}, metastases: {met}")
#     print(f"Actual - fracture: {actual_f}, metastases: {actual_m}")


#     report = """
# Abdomen + Pelvis CT without oral C+ CT ABDOMEN AND PELVIS:

# Clinical history: Prostate cancer and bone metastases. Chemotherapy for one week PTA. Now with fever and lower abdominal pain query neutropenic colitis.

# Comparison exam:None available.

# Axial volumetric CT images were obtained from above the level of the diaphragms to below the level of the ischial tuberosities following the use of intravenous and oral contrast with 5 mm slice thickness.

# Findings:

# There is significant thickening of the sigmoid colon with a defect demonstrated along the posterior aspect of the sigmoid wall which leads into a region of fatty stranding and extraluminal gas consistent with a perforated diverticulitis. No abscess
# appreciated.
# Multiple sigmoid colon diverticuli are noted. No evidence of bowel obstruction.

# There is fatty infiltration of the colon seen most conspicuously within the ascending, transverse, and descending portions. This is nonspecific but can be seen in the context of past inflammatory bowel disease.

# The stomach and small bowel are unremarkable.

# Hypodensity within segment 4B liver measuring 0.8 cm in size is most likely a cyst.
# The spleen and biliary tree are unremarkable.
# The gallbladder has been removed.

# The adrenal glands and kidneys are unremarkable.

# Bladder appears mildly thickened but is incompletely distended.

# Multiple para-aortic lymph nodes are seen none of which meet CT criteria for pathologic enlargement. Largest measures 0.9 cm.

# Fat-containing umbilical hernia. Right-sided inguinal hernia repair noted.

# Suspected metastatic disease to the bones demonstrated on prior bone scan is not well seen on today's study. There is a small sclerotic focus noted in the left iliac crest (image 89) which may correlate with the nuclear medicine bone scan of January
# 28, 2016.
#  Compression fracture of T12 noted. Endplate fractures of L2 and L3 also demonstrated. There are degenerative changes of the lumbar spine.

# IMPRESSION:

# Findings in keeping with sigmoid diverticulitis with a small focal perforation. No associated abscess.


# _____________

# [DEIDENTIFIED - DOCTOR'S INFO]

# [DEIDENTIFIED - DOCTOR'S INFO]
# Report from 2016-04-25 00:00:00: Abdomen + Pelvis CT with oral C+ CT ABDOMEN PELVIS (ENHANCED)

# HISTORY: 56 yo Male. prostate - high risk, likely M1. cT3b, gleason 10 (11/11 cores), widespread bone mets - ad re-staging following initiation of ADT and 6xdocetaxel chemotherapy before consideration of radiotherapy

# COMPARISON: whole-body bone scan dated August 18, 2016. CT dated April 25, 2016.

# TECHNIQUE: CT images of the abdomen and pelvis with oral and intravenous contrast.

# FINDINGS:  Prostate not particularly enlarged by CT, measuring 3.4 cm (previous 3.5 cm). No gross abnormality of the prostate by CT. No evidence for involvement of adjacent structures.

# Stable mild bladder wall thickening. No hydronephrosis or hydroureter.

# No enlarged lymph nodes. No ascites. No peritoneal deposit.

# Stable appearance to the bones, including metastatic involvement of the left iliac crest.

# No evidence of liver metastases. Stable hypodensity in the liver segment 4B. Cholecystectomy.

# Evolution of the previous acute sigmoid diverticulitis. Small focus of contained extraluminal gas again present, now better circumscribed, measuring (image 125) 2.3 cm (previous 2.5 cm). The position of this contained gas has shifted slightly, and
# is now intimately associated with the bladder wall. No evidence for fistula to the bladder lumen. Associated inflammatory fat stranding is similar to slightly decreased.

# Sigmoid wall thickening seen in this region, likely related to be diverticulitis; correlation with non-urgent colonoscopy recommended once the patient's acute condition has resolved.

# No other change.

# CT chest reported separately.

# SUMMARY:

# Stable bony metastatic disease.

# Evolution of the acute sigmoid diverticulitis, as described above. Slight improvement is seen, however the change is quite minimal considering it has been approximately a 5 month since the prior study. Clinical correlation required.

# Sigmoid wall thickening, likely related to the diverticulitis; correlation with non-urgent colonoscopy recommended once the patient's acute condition has resolved.
# """

#     frac, met = predict_label_lama(report)
#     actual_f, actual_m = 0, 1
#     print(f"Predicted - fracture: {frac}, metastases: {met}")
#     print(f"Actual - fracture: {actual_f}, metastases: {actual_m}")

    report = """
"Bone Scan(Whole Body)Nuc Med TECHNETIUM MDP BONE SCAN

Findings:

There is a focus of intense activity involving the T6 vertebral body in the midline, concerning for metastasis.

There is focal activity involving the left eighth rib laterally.  When correlated with abdominal/pelvic CT of the same day there is subtle sclerosis in this region without a discrete lesion, and no evidence of fracture.

There is activity in the lower cervical spine at the cervicothoracic junction on the right, most commonly degenerative. There are small foci of activity at the left T10 costovertebral junction and in the left aspect of T12.  When correlated with CT
of the same day the changes may be degenerative.

INTERPRETATION:

The lesion at T6 is highly concerning for metastasis in this context.  Suggest correlation with CT through this region.  The lesion involving the left eighth rib laterally is suspicious for metastasis as well.
_____________

[DEIDENTIFIED - DOCTOR'S INFO] X-Ray Chest PA+LAT Routine CHEST PA AND LATERAL

Reference:No previous

The cardiac silhouette, mediastinal and hilar structures are within normal limits.  The lungs and pleural spaces are clear.
_____________

[DEIDENTIFIED - DOCTOR'S INFO]
Report from 2011-02-02 00:00:00: Abdomen + Pelvis CT with oral C+ CT ABDOMEN PELVIS (ENHANCED)

COMPARISON: CT dated February 2 2011.  Spine MR dated March 8 2011.  Whole body bone scan dated April 11 2011, the same day as the CT.

TECHNIQUE: CT images of the abdomen and pelvis with oral and intravenous contrast.

FINDINGS: Total prostatectomy.  Stable appearance to the prostate bed, with no gross abnormality.  Surgical dissection clips noted.

Mottled lucency of sclerosis in left eighth rib laterally, correlating to increased activity on recent bone scan (which had progressed since the bone scan prior).  Bony metastasis has also been demonstrated in the T6 vertebral body, on bone scan and
MRI, but is not imaged on this CT. MRI also demonstrated metastatic involvement of the left T12 pedicle, with no definite corresponding finding on CT.

No lymphadenopathy.

Stable hypodense lesion in the right kidney, likely a cyst.  Abdominal pelvic visceral otherwise unremarkable.

No concerning finding in the lung bases.

INTERPRETATION:

Bony metastases, better assessed on bone scan and MRI.

No other sites of metastatic disease.




_____________

[DEIDENTIFIED - DOCTOR'S INFO]
Report from 2011-04-11 00:00:00: Abdomen + Pelvis CT with oral C+ CT ABDOMEN PELVIS (ENHANCED)

COMPARISON: CT dated April 11 and February 2 2011.  Spine MR dated March 8 2011.  Same Day whole body bone scan and prior bone scan dated April 11 2011.

TECHNIQUE: CT images of the abdomen and pelvis with oral and intravenous contrast.

FINDINGS: Total prostatectomy.  Stable appearance to the prostate bed, with no gross abnormality.  Surgical dissection clips noted.

Bony metastases have been demonstrated in the left eighth rib and T6 vertebral body, on bone scan and MRI, but not imaged on this CT. MRI also demonstrated metastatic involvement of the left T12 pedicle, with no definite corresponding finding on CT.

No lymphadenopathy.

New ovoid low-density (fluid density) structure measuring 1.6 x 0.7 cm in the right flank subcutaneous fat, atypical for a prostate metastasis, of uncertain etiology but likely incidental.

Stable hypodense lesion in the right kidney, likely a cyst.  Abdominal pelvic viscera otherwise unremarkable.

No concerning finding in the lung bases.

INTERPRETATION:

Bony metastases, better assessed on bone scan and MRI.

No other definite sites of metastatic disease.

New ovoid low-density (fluid density) structure measuring 1.6 x 0.7 cm in the right flank subcutaneous fat, atypical for a prostate metastasis, of uncertain etiology but likely incidental.  This can be reassessed on follow up imaging.

_____________

[DEIDENTIFIED - DOCTOR'S INFO] Bone Scan(Whole Body)Nuc Med TECHNETIUM MDP BONE SCAN

Comparison: 4/11/11

Lesions are again seen at the cervicothoracic junction on the right, T6 diffusely, T12 on the left, and the left eighth rib laterally, in keeping with metastases.  There are no new lesions seen.  No significant change.
_____________

[DEIDENTIFIED - DOCTOR'S INFO]
Report from 2011-06-06 00:00:00: Bone Scan(Whole Body)Nuc Med TECHNETIUM MDP BONE SCAN

Comparison: June 6, 2011

Previously seen lesions at the cervicothoracic junction, T6, T12, and left eighth rib laterally appear stable.  There is a tiny focus of activity at the superior aspect of the right greater trochanter, more conspicuous than previous but in
retrospect likely present on the prior studies as well.  It is nonspecific possibly representing mild enthesopathy, and can be reassessed at follow-up.  No significant change otherwise.
_____________

[DEIDENTIFIED - DOCTOR'S INFO]"
"""