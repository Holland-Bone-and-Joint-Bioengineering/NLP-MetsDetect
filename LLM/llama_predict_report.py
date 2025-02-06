import transformers
import torch
import os
import pandas as pd 

from few_shot_examples import ex3, ex4, ex2_met, fracture_examples, metastases_examples

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model()
    # Load the model and tokenizer
    print("Loading model ...")

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    return pipeline

def predict_label_lama(pipeline, report):
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


def process_report_csv(csv_file_name):
    df_reports = pd.read_csv(csv_file_name)

    df_reports["report_and_frac_label"] = (
        "Report:\n" + 
        df_reports["combined_reports"] + 
        "\n\nFracture classification:\n" + 
        df_reports["image_ct___1"]
        # df_reports["image_ct___1"].apply(lambda x: "Positive" if float(x) > 0 else "Negative")
    )

    df_reports["report_and_mets_label"] = (
        "Report:\n" + 
        df_reports["combined_reports"] + 
        "\n\nMetastases classification:\n" + 
        df_reports["image_ct___1"]
        # df_reports["image_ct___2"].apply(lambda x: "Positive" if float(x) > 0 else "Negative")
    )

    # drop reports that have NaN in reports column
    df_reports = df_reports.dropna(subset=["report_and_frac_label", "report_and_mets_label"])

    texts = df_reports['combined_reports'].tolist()
    frac_labels = df_reports['image_ct___1'].astype(float).round().astype(int).values.tolist()
    mets_labels = df_reports['image_ct___2'].astype(float).round().astype(int).values.tolist()

    return texts, frac_labels, mets_labels



if __name__ == "__main__":
    pipeline = load_model()

    texts, frac_labels, mets_labels = process_report_csv('data/labeled_data_combined_reports.csv')

    i = 0
    for report, label_frac, label_met in zip(texts, frac_labels, mets_labels):
        print(f"\nPredicting report {i}")
        frac, met = predict_label_lama(pipeline, report)
        print(f"Predicted - fracture: {frac}, metastases: {met}")
        print(f"Actual - fracture: {label_frac}, metastases: {label_met}")
        i += 1