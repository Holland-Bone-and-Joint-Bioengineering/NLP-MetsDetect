import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


### FEW SHOT LLAMA WITH:
    # - separate classifications for metastases and fractures
    # - using the combined reports
    # trying to minimize false negative rate

### TO USE:
    # 1) get access to LLAMA from hugging face page
    # 2) get a token (see instructions online)
    # 3) run `git config --global credential.helper store`
    # 4) run `huggingface-cli login` and then enter your token
    # 5) then you can just run this script, `python3 fewshot_llama_attempt2.py`


# Define model path
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Use instruction-tuned model
cache_dir = "/home/nsathish/.cache/huggingface/hub"

# Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained(model_name) #, cache_dir=cache_dir, legacy=False)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto") #, cache_dir=cache_dir)

# Few-shot examples
few_shot_examples = """
Does this radiology report text indicate fracture and/or metastases? Provide an answer using the example output format below.
- Example Input: "PROVIDED HISTORY: "NONE, Prostate ca. follow-up after SBRT to oligomets ???Please do RECIST 1.1 and PCWG3 for PR.20 study (TIMEPOINT-4FINDINGS: Skeletal phase whole body planar and tomographic imaging of the thorax/abdomen/pelvis/proximal femurs demonstrate stable scislightly less conspicuous tracer activity within the mid thoracic spine. Otherwise, there is no scintigraphic abnormality suspicious ocorresponding to insufficiency fractures on recent MRI. There is degenerative-appearing uptake within the left lower cervical spine, s[DEIDENTIFIED - DOCTOR'S INFO]ence for disease progression as per PCWG-3."
- Example Output: "Metastases: Yes Fracture: Yes"

Does this radiology report text indicate fracture and/or metastases? 
- Example Input: "Provided History: "T2b (MR nodule centrally 1.7cm) G6 19.6 DT1.6y velocity 7.2 /yr. Pt on Abiraterone+/-Ipateserib. Please compare ALLFindings: Angiographic and tissue phase imaging of the torso is unremarkable. Skeletal phase whole body planar images and SPECT of thedegenerative-appearing uptake within the right a.c. joint, right L4/L5 facet joint, knees and forefeet. Stable mild diffuse tracer act[DEIDENTIFIED - DOCTOR'S INFO]ence for bony metastatic disease.remote trauma."
- Example Output: "Metastases: No Fracture: No"
"""

# Define prediction function
def predict_label_llama(report, few_shot_examples, model, tokenizer):
    prompt = few_shot_examples + f'- Input: "{report}"\n- Output: '
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"].to("cuda"),
        max_length=20000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract prediction ("Metastases" or "Fracture")
    if "Metastases" in response:
        return 1  # Assuming 1 corresponds to Metastases
    elif "Fracture" in response:
        return 0  # Assuming 0 corresponds to Fracture
    else:
        return -1  # Handle unclear cases

# Load labeled data FROM COMBINED REPORTS
print("Loading labeled data...")
grouped_reports = pd.read_csv('data/labeled_data_combined_reports.csv')  # Replace with the correct path

grouped_reports['image_ct___1'] = pd.to_numeric(grouped_reports['image_ct___1'], errors='coerce')
grouped_reports['image_ct___2'] = pd.to_numeric(grouped_reports['image_ct___2'], errors='coerce')

grouped_reports = grouped_reports.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=['image_ct___1', 'image_ct___2'])

grouped_reports['image_ct___1'] = grouped_reports['image_ct___1'].astype(int)
grouped_reports['image_ct___2'] = grouped_reports['image_ct___2'].astype(int)

texts = grouped_reports['combined_reports'].tolist()
labels = grouped_reports[['image_ct___1', 'image_ct___2']].values.tolist()

# Split into train/test sets (optional, if not already split)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# use the first train data text and the second train data text for the instructions
example1 = train_texts[0]
example1_label = train_labels[0]

example2 = train_texts[1]
example2_label = train_labels[1]




# Generate predictions for the test set
print("Generating predictions...")
predictions = []
true_labels = []

i = 0
for report, label in zip(test_texts, test_labels):
    print("Prediction: " + str(i))
    prediction = predict_label_llama(report, few_shot_examples, model, tokenizer)
    predictions.append(prediction)
    true_labels.append(label)
    print("\nPredictions: " + str(predictions))
    print("\nTrue labels: " + str(true_labels))
    i = i + 1
    if (i == 3):
        break

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
