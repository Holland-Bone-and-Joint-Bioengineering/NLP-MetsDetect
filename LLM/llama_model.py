from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = "/home/nsathish/.llama/checkpoints/Llama3.1-8B-Instruct/"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)

text = "The radiology report indicates significant changes in the bone structure. Does it suggest a fracture or metastases?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

