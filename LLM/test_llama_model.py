from transformers import LlamaForCausalLM, LlamaTokenizerFast
import os

# Define model path
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Use instruction-tuned model
cache_dir = "/home/nsathish/.cache/huggingface/hub"

# Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir, legacy=False)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=cache_dir)

# Test the model with a sample prompt
prompt = (
    "The radiology report mentions a lesion in the L3 vertebra. "
    "Does this suggest a fracture or metastases? Please explain in detail."
)

inputs = tokenizer(prompt, return_tensors="pt")
print("Generating text...")
outputs = model.generate(
    inputs["input_ids"].to("cuda"), 
    max_length=200, 
    temperature=0.7, 
    top_p=0.9, 
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Response:")
print(response)

