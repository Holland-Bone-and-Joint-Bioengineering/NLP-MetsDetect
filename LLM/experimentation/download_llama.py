from transformers import LlamaForCausalLM, LlamaTokenizer

# Define the repository name
model_name = "meta-llama/Llama-3.1-8B"

# Download the model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=True)

# Save the model locally
model.save_pretrained("/home/nsathish/llama/llama-3.1-8b")
tokenizer.save_pretrained("/home/nsathish/llama/llama-3.1-8b")
print("Model and tokenizer downloaded successfully!")
