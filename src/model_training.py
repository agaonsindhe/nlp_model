import torch
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
from src.data_processing import load_and_preprocess_dataset, tokenize_dataset

def fine_tune_t5(file_path, output_dir, model_name="google/t5-base"):
    """Fine-tune T5 model on the dataset."""
    # Load tokenizer and dataset
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    inputs, targets = load_and_preprocess_dataset(file_path)
    tokenized_data = tokenize_dataset(inputs, targets, tokenizer)

    # Create a PyTorch Dataset
    class GrammarDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
                "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
                "labels": torch.tensor(self.encodings["labels"][idx]),
            }

    dataset = GrammarDataset(tokenized_data)

    # Load the model
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_steps=500,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
