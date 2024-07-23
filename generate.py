from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_gpt2')

def generate_script(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "FADE IN: INT. COFFEE SHOP - DAY\n\nJohn and Sarah sit across from each other, sipping their drinks."
    generated_script = generate_script(prompt)
    print(generated_script)
