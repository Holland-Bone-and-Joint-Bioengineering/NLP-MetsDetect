import transformers
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load the model and tokenizer
print("Loading model ...")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Few-shot examples

ex3 = """
Bone Scan(Whole Body)Nuc Med TECHNETIUM MDP BONE SCAN WHOLE BODY:

History:Prostate cancer

Comparison:CT chest, abdomen and pelvis 9/21/10. No previous bone
scans

Findings:Static whole body images show foci of increased activity
in the left glenoid, T11, L2, the left medial iliac bone, the left
proximal femur and the right acetabulum. These findings are in
keeping with bony metastases.

Summary:Bony metastases as described above.

[DEIDENTIFIED - DOCTOR'S INFO]
"""

ex4 = """
Bone Scan Whole Body+Extra Views+Flow+Spect TECHNETIUM MDP BONE SCAN WHOLE BODY: SPECT lumbosacral spine and pelvis

History:Prostate cancer.  Rapidly rising PSA

Comparison:Bone scan 3/16/11

Findings:Posterior flow study and blood pool images of the pelvis demonstrate questionable mild hyperemia in the right iliac crest.

Static images demonstrate increased activity superolaterally in the right iliac wing of the .  This has increased in size and intensity since bone scan of 3/16/11 and is consistent with bony metastasis until otherwise proven.  Mild increased
activity at L4 for corresponds to degenerative change on CT.  Activity just inferior to the right 11th rib probably related to mild retention of radiopharmaceutical in a calyx in the right mid kidney related to a mid right renal stone.  Activity in
both knees is in keeping with arthritis.

Summary:Activity in right iliac wing consistent with bony metastasis with progression since bone scan of 3/16/11
_____________

[DEIDENTIFIED - DOCTOR'S INFO]
"""

ex2_met = f"""
Bone Scan(Whole Body)Nuc Med WHOLE BODY BONE SCAN

Clinical History: Prostate cancer, restaging.

Reference examination: Multiple prior bone scans most recently
from March 2007. Prior CT abdomen from March 2007.

Findings:

Mildly inhomogeneous activity is again noted within the ribs, with
more focality right posterolateral 10th rib. These are unchanged.

Inhomogeneity lower cervical, mid and lower thoracic, and upper to
mid lumbar spine, likely degenerative/arthritic.

Interpretation:

No clear-cut scintigraphic evidence for skeletal metastases. No
significant interval change.
"""

fracture_examples = f"""
Does this radiology report text indicate fracture in patient's bones? Provide an answer using the example output format below. Make sure to adhere to the strict format one of the two options: "Fracture: Yes", "Fracture: No".

Respond with "No" only if you are decently confident that there is no fracture. Respond with "Yes" only if you are decently confident that there is no fracture. If you are uncertain, or if the report does not clearly indicate either way, respond with "Unknown".

Example 1. Does this radiology report text indicate fracture?
- Example Input: "PROVIDED HISTORY: "NONE, Prostate ca. follow-up after SBRT to oligomets ???Please do RECIST 1.1 and PCWG3 for PR.20 study (TIMEPOINT-4FINDINGS: Skeletal phase whole body planar and tomographic imaging of the thorax/abdomen/pelvis/proximal femurs demonstrate stable scislightly less conspicuous tracer activity within the mid thoracic spine. Otherwise, there is no scintigraphic abnormality suspicious ocorresponding to insufficiency fractures on recent MRI. There is degenerative-appearing uptake within the left lower cervical spine, s[DEIDENTIFIED - DOCTOR'S INFO]ence for disease progression as per PCWG-3."
- Example Output: "Fracture: Yes"
Reasoning: the report talks about the existence of "insufficiency fractures".

Example 2. Does this radiology report text indicate fracture?
- Example Input: "{ex3}"
- Example Output: "Fracture: Unknown"
Reasoning: the report does not discuss bone fractures.

Example 3. Does this radiology report text indicate fracture?
- Example Input: "{ex4}"
- Example Output: "Fracture: Unknown"
Reasoning: the report is a bone scan but does not discuss bone fractures.
"""


metastases_examples = f"""
Does this radiology report text indicate metastases or metastatic diseases? Provide an answer using the example output format below. Make sure to adhere to the strict format one of the two options: "Metastases: Yes", "Metastases: No".

Respond with "No" only if you are decently confident that there is no metastases. Respond with "Yes" only if you are decently confident that there is no metastases. If you are uncertain, or if the report does not clearly indicate either way, respond with "Unknown".

Example 1. Does this radiology report text indicate metastases?
- Example Input: "PROVIDED HISTORY: "NONE, Prostate ca. follow-up after SBRT to oligomets ???Please do RECIST 1.1 and PCWG3 for PR.20 study (TIMEPOINT-4FINDINGS: Skeletal phase whole body planar and tomographic imaging of the thorax/abdomen/pelvis/proximal femurs demonstrate stable scislightly less conspicuous tracer activity within the mid thoracic spine. Otherwise, there is no scintigraphic abnormality suspicious ocorresponding to insufficiency fractures on recent MRI. There is degenerative-appearing uptake within the left lower cervical spine, s[DEIDENTIFIED - DOCTOR'S INFO]ence for disease progression as per PCWG-3."
- Example Output: "Metastases: Yes"
Reasoning:
From the phrase "Prostate ca. follow-up after SBRT to oligomets": SBRT (Stereotactic Body Radiation Therapy) is commonly used to treat small, localized metastatic lesions (referred to as oligometastases).
The phrase "follow-up after SBRT to oligomets" confirms that the patient had metastases in the past, which were treated with SBRT.

Also, from the phrase "Purpose of Imaging ("RECIST 1.1 and PCWG-3 for PR.20 study")":
These criteria are used for evaluating cancer treatment response and progression, further supporting that the patient had prior metastatic disease.

Example 2. Does this radiology report text indicate metastases?
- Example Input: "{ex2_met}"
- Example Output: "Metastases: No"
Reasoning: the report says "No clear-cut scintigraphic evidence for skeletal metastases."

Example 3. Does this radiology report text indicate metastases?
- Example Input: "{ex4}"
- Example Output: "Metastases: Yes"
Reasoning: the report indicates "is consistent with bony metastasis until otherwise proven."
"""

def predict_label_lama(report):
    frac_prompt = fracture_examples + f'Now classify the following report and provide a brief explanation.\n- Example Input: "{report}"\n- Example Output: '
    met_prompt = metastases_examples + f'Now classify the following report and provide a brief explanation.\n- Example Input: "{report}"\n- Example Output: '
    res = []
    for prompt in [frac_prompt, met_prompt]:
        messages = [
            {"role": "system", "content": "You are an intelligent assistant who's tasked with identifying whether a radiology report describes a paitient with fractures and/or metastases."},
            {"role": "user", "content": prompt},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=2048,
        )

        response = outputs[0]["generated_text"][-1]

        processed_response = response["content"]
        print("***** Full Response *****")
        print(processed_response)

        if "Metastases" in processed_response:
            if "Yes" in processed_response:
                res.append(1)
            elif "No" in processed_response:
                res.append(0)
            elif "Unknown" in processed_response:
                res.append(0.5) # assume no fracture/met when unknown (Note this could lead to some false positives but this is the best educated guess)
            else:
                res.append(-1)
        elif "Fracture" in processed_response:
            if "Yes" in processed_response:
                res.append(1)
            elif "No" in processed_response:
                res.append(0)
            elif "Unknown" in processed_response:
                res.append(0.5)
            else:
                res.append(-1)
        else:
            res.append(-1)

    return res


if __name__ == "__main__":
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


    report = """
Abdomen + Pelvis CT without oral C+ CT ABDOMEN AND PELVIS:

Clinical history: Prostate cancer and bone metastases. Chemotherapy for one week PTA. Now with fever and lower abdominal pain query neutropenic colitis.

Comparison exam:None available.

Axial volumetric CT images were obtained from above the level of the diaphragms to below the level of the ischial tuberosities following the use of intravenous and oral contrast with 5 mm slice thickness.

Findings:

There is significant thickening of the sigmoid colon with a defect demonstrated along the posterior aspect of the sigmoid wall which leads into a region of fatty stranding and extraluminal gas consistent with a perforated diverticulitis. No abscess
appreciated.
Multiple sigmoid colon diverticuli are noted. No evidence of bowel obstruction.

There is fatty infiltration of the colon seen most conspicuously within the ascending, transverse, and descending portions. This is nonspecific but can be seen in the context of past inflammatory bowel disease.

The stomach and small bowel are unremarkable.

Hypodensity within segment 4B liver measuring 0.8 cm in size is most likely a cyst.
The spleen and biliary tree are unremarkable.
The gallbladder has been removed.

The adrenal glands and kidneys are unremarkable.

Bladder appears mildly thickened but is incompletely distended.

Multiple para-aortic lymph nodes are seen none of which meet CT criteria for pathologic enlargement. Largest measures 0.9 cm.

Fat-containing umbilical hernia. Right-sided inguinal hernia repair noted.

Suspected metastatic disease to the bones demonstrated on prior bone scan is not well seen on today's study. There is a small sclerotic focus noted in the left iliac crest (image 89) which may correlate with the nuclear medicine bone scan of January
28, 2016.
 Compression fracture of T12 noted. Endplate fractures of L2 and L3 also demonstrated. There are degenerative changes of the lumbar spine.

IMPRESSION:

Findings in keeping with sigmoid diverticulitis with a small focal perforation. No associated abscess.


_____________

[DEIDENTIFIED - DOCTOR'S INFO]

[DEIDENTIFIED - DOCTOR'S INFO]
Report from 2016-04-25 00:00:00: Abdomen + Pelvis CT with oral C+ CT ABDOMEN PELVIS (ENHANCED)

HISTORY: 56 yo Male. prostate - high risk, likely M1. cT3b, gleason 10 (11/11 cores), widespread bone mets - ad re-staging following initiation of ADT and 6xdocetaxel chemotherapy before consideration of radiotherapy

COMPARISON: whole-body bone scan dated August 18, 2016. CT dated April 25, 2016.

TECHNIQUE: CT images of the abdomen and pelvis with oral and intravenous contrast.

FINDINGS:  Prostate not particularly enlarged by CT, measuring 3.4 cm (previous 3.5 cm). No gross abnormality of the prostate by CT. No evidence for involvement of adjacent structures.

Stable mild bladder wall thickening. No hydronephrosis or hydroureter.

No enlarged lymph nodes. No ascites. No peritoneal deposit.

Stable appearance to the bones, including metastatic involvement of the left iliac crest.

No evidence of liver metastases. Stable hypodensity in the liver segment 4B. Cholecystectomy.

Evolution of the previous acute sigmoid diverticulitis. Small focus of contained extraluminal gas again present, now better circumscribed, measuring (image 125) 2.3 cm (previous 2.5 cm). The position of this contained gas has shifted slightly, and
is now intimately associated with the bladder wall. No evidence for fistula to the bladder lumen. Associated inflammatory fat stranding is similar to slightly decreased.

Sigmoid wall thickening seen in this region, likely related to be diverticulitis; correlation with non-urgent colonoscopy recommended once the patient's acute condition has resolved.

No other change.

CT chest reported separately.

SUMMARY:

Stable bony metastatic disease.

Evolution of the acute sigmoid diverticulitis, as described above. Slight improvement is seen, however the change is quite minimal considering it has been approximately a 5 month since the prior study. Clinical correlation required.

Sigmoid wall thickening, likely related to the diverticulitis; correlation with non-urgent colonoscopy recommended once the patient's acute condition has resolved.
"""

    frac, met = predict_label_lama(report)
    actual_f, actual_m = 0, 1
    print(f"Predicted - fracture: {frac}, metastases: {met}")
    print(f"Actual - fracture: {actual_f}, metastases: {actual_m}")