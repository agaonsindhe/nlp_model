from transformers import T5Tokenizer, T5ForConditionalGeneration

def test_model(sentence, model_path="models/fine_tuned_t5_base"):
    """Test the fine-tuned model with a given sentence."""
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    input_text = f"grammar correction: {sentence}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)

    outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
