from transformers import MarianMTModel, MarianTokenizer

def kannada_to_english(text):
    model_name = "Helsinki-NLP/opus-mt-kn-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    output = model.generate(**inputs)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

sentence = input("Enter Kannada sentence: ")
print("English Translation:", kannada_to_english(sentence))
