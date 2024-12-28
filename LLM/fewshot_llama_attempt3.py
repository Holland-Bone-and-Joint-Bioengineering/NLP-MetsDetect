import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import transformers
from sklearn.model_selection import train_test_split
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
Reasoning; the report indicates "is consistent with bony metastasis until otherwise proven."
"""

def predict_label_lama(report):
    frac_prompt = fracture_examples + f'- Final Input: "{report}"\n- Final Output: '
    met_prompt = metastases_examples + f'- Final Input: "{report}"\n- Final Output: '
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

        processed_response = response["content"].split("Final Output: ")[-1]
        print("Processed response:", processed_response)

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

# Load labeled data
print("Loading labeled data...")
grouped_reports = pd.read_csv('../../LLM/labeled_data_combined_reports.csv')
# grouped_reports = pd.read_csv('../../LLM/labeled_data.csv')

grouped_reports['image_ct___1'] = pd.to_numeric(grouped_reports['image_ct___1'], errors='coerce')
grouped_reports['image_ct___2'] = pd.to_numeric(grouped_reports['image_ct___2'], errors='coerce')

grouped_reports = grouped_reports.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=['image_ct___1', 'image_ct___2'])

grouped_reports['image_ct___1'] = grouped_reports['image_ct___1'].astype(int)
grouped_reports['image_ct___2'] = grouped_reports['image_ct___2'].astype(int)

texts = grouped_reports['combined_reports'].tolist()

labels = grouped_reports[['image_ct___1', 'image_ct___2']].values.tolist()

############### CLASSIFYING FRACTURES ##################
# Split into train/test sets (optional, if not already split)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# Generate predictions for the test set
print("Generating predictions...")
predictions = []
true_labels = []

print("Total examples:", len(test_texts))
i = 0
for report, label in zip(test_texts, test_labels):
    print("Report length:", len(report))
    # Truncate if necessary
    MAX_LEN = 10000
    if len(report) > MAX_LEN:
        report = report[len(report)-MAX_LEN:]
    print("Prediction: " + str(i))
    prediction = predict_label_lama(report)
    predictions.append(prediction)
    true_labels.append(label)
    print("\nPredictions: " + str(predictions))
    print("\nTrue labels: " + str(true_labels))
    i = i + 1

# Convert lists to arrays
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Filter out invalid predictions (-1)
valid_indices = predictions != -1
valid_predictions = predictions[valid_indices]
valid_true_labels = true_labels[valid_indices]

# Evaluate metrics
print("Evaluating performance...")
accuracy = accuracy_score(valid_true_labels, valid_predictions)
f1 = f1_score(valid_true_labels, valid_predictions, average="binary")  # Adjust as needed

# Per-label accuracy
unique_labels = np.unique(valid_true_labels)
per_label_accuracy = {
    label: accuracy_score(
        valid_true_labels[valid_true_labels == label],
        valid_predictions[valid_true_labels == label],
    )
    for label in unique_labels
}

# Print results
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
for label, acc in per_label_accuracy.items():
    label_name = "Metastases" if label == 1 else "Fracture"
    print(f"Accuracy for {label_name}: {acc:.4f}")