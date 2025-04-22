import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from sklearn.model_selection import train_test_split
import os
import torch
from datetime import datetime
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define model path
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Use instruction-tuned model
cache_dir = "/home/salnassa/.cache/huggingface/hub"

# Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir, legacy=False)
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix for padding issue
model = model = LlamaForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"  # Let HF place model across CPU+GPU if needed
).to("cuda") 

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
def predict_label_lama(report, few_shot_examples, model, tokenizer):
    prompt = few_shot_examples + f'- Input: "{report}"\n- Output: '
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"].to("cuda"),
        attention_mask=inputs["attention_mask"].to("cuda"),
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "- Output:" in full_text:
        generated_part = full_text.split("- Output:")[-1].strip()
    else:
        generated_part = full_text.strip()
    print("Generated Response:\n", generated_part)
    # Parse both labels from response
    met, frac = -1, -1
    generated_lower = generated_part.lower()
    if "metastases: yes" in generated_lower:
        met = 1
    elif "metastases: no" in generated_lower:
        met = 0

    if "fracture: yes" in generated_lower:
        frac = 1
    elif "fracture: no" in generated_lower:
        frac = 0

    return frac, met

def evaluate_task(y_true, y_pred):
    '''
    Evaluate binary classification performance with detailed metrics and error inspection for fracture and metastases predictions

    Args:
        y_true (array-like): Ground truth labels (expected to be binary: 0 or 1).
        y_pred (array-like): Predicted labels from the model (expected to be binary: 0 or 1).

    Returns:
        None. Prints the following evaluation metrics:
            - Accuracy
            - Precision
            - Recall
            - F1 Score
            - Confusion matrix components (TP, FP, TN, FN)
            - Indices of false negatives for inspection
    '''
    
    # Convert to Series and make sure values are numeric
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    # Keep only rows where both are exactly 0 or 1
    mask = y_true.isin([0, 1]) & y_pred.isin([0, 1])
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)

    # Now compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    print(f"False Negatives (FN) at indices: {false_negatives.tolist()}")
    

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Load labeled data
print("Loading labeled data...")
grouped_reports = pd.read_csv('/home/salnassa/capstone/data/grouped_test.csv')  # Replace with the correct path
texts = grouped_reports['reports'].tolist()
labels = grouped_reports[['image_ct___1', 'image_ct___2']].astype(int).values.tolist()

# Generate predictions for the test set
print("Generating predictions...")
predictions = []
true_labels = []

timestamp = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm')
csv_filename = f"results/fewshot_test_{timestamp}.csv"

#Write outputs to csv
with open(csv_filename, mode='w', newline='') as file:
    print(f"Writing to {csv_filename}")
    writer = csv.writer(file)
    
    # write headers
    writer.writerow(['mrn', 'imaging_date', 'study_uid', 'Predicted_frac', 'Predicted_mets', 'Actual_frac', 'Actual_mets'])
    i = 0
    #Iterate through reports and labels 
    for report, label, mrn, date, study_uid in zip(
        texts,
        labels,
        grouped_reports['patient_id'],
        grouped_reports['assessment_date'],
        grouped_reports['study_uid']
    ):
        print("Prediction: " + str(i))
        MAX_RETRIES = 3
        retries = 0
        while retries < MAX_RETRIES:
            #Predict fracture and metastases 
            frac_pred, met_pred = predict_label_lama(report, few_shot_examples, model, tokenizer)
            if frac_pred != -1 and met_pred != -1:
                break
            print(f"Retry {retries+1} due to -1 prediction...")
            retries += 1

        #Append predictions to list 
        predictions.append([frac_pred, met_pred])
        true_labels.append(label)
        print("\nPredictions: " + str(predictions))
        print("\nTrue labels: " + str(true_labels))
        #Write out csv row with all information 
        writer.writerow([mrn, date, study_uid, frac_pred, met_pred, label[0], label[1]])
        i +=1



predictions = np.array(predictions)
true_labels = np.array(true_labels)

#Use only valid predictions
valid_mask = ~np.all(predictions == -1, axis=1)

#Define fracture and metastases predictions and labels 
frac_pred = predictions[valid_mask][:, 0]
mets_pred = predictions[valid_mask][:, 1]
frac_true = true_labels[valid_mask][:, 0]
mets_true = true_labels[valid_mask][:, 1]

#Evaluate performances for fracture and Metastases
print("Evaluating performance Fracture...")
evaluate_task(frac_true,frac_pred)

print("Evaluating performance Mets...")
evaluate_task(mets_true, mets_pred)