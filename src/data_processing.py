import tensorflow as tf
from transformers import T5Tokenizer

# Define the feature description
FEATURE_DESCRIPTION = {
    "input_text": tf.io.FixedLenFeature([], tf.string),
    "target_text": tf.io.FixedLenFeature([], tf.string),
}

def parse_tfrecord(example_proto):
    """Parse a single TFRecord example."""
    return tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

def load_and_preprocess_dataset(file_path):
    """Load and preprocess the dataset."""
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord)

    inputs, targets = [], []
    for record in parsed_dataset:
        inputs.append(record["input_text"].numpy().decode("utf-8"))
        targets.append(record["target_text"].numpy().decode("utf-8"))

    return inputs, targets

def tokenize_dataset(inputs, targets, tokenizer, max_length=128):
    """Tokenize the dataset for T5."""
    input_texts = [f"grammar correction: {text}" for text in inputs]
    encodings = tokenizer(input_texts, max_length=max_length, padding="max_length", truncation=True)
    target_encodings = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
    encodings["labels"] = target_encodings["input_ids"]
    return encodings
