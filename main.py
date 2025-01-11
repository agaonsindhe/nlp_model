import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import joblib  # For saving scaler and tokenizer

# Set Paths
DATA_DIR = "./data"  # Directory containing all TFRecord files
MODEL_DIR = "./models"
TFRECORD_PATTERN = os.path.join(DATA_DIR, "c4200m-train.tfrecord-*")  # Pattern to match all TFRecord files
MAX_SEQUENCE_LENGTH = 100  # Define the maximum sequence length

# Parse TFRecords
def parse_tfrecord(example):
    feature_description = {
        "input": tf.io.FixedLenFeature([], tf.string),
        "output": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    return parsed_example["input"].numpy().decode("utf-8"), parsed_example["output"].numpy().decode("utf-8")

# Load and Preprocess Data
def load_tfrecord_dataset(file_pattern):
    print(f"Matching files with pattern: {file_pattern}")
    files = tf.io.matching_files(file_pattern)

    if tf.size(files).numpy() == 0:
        print("No TFRecord files found. Check your DATA_DIR and file naming.")
        return [], []

    print(f"Found {len(files)} files.")
    raw_dataset = tf.data.TFRecordDataset(files)

    inputs, outputs = [], []
    for raw_record in raw_dataset:
        try:
            parsed_example = parse_tfrecord(raw_record)
            if parsed_example[0] and parsed_example[1]:
                inputs.append(parsed_example[0])
                outputs.append(parsed_example[1])
        except Exception as e:
            print(f"Error parsing record: {e}")

    print(f"Loaded {len(inputs)} input-output pairs.")
    return inputs, outputs

def tokenize_and_preprocess(inputs, outputs, tokenizer=None):
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(inputs + outputs)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(tokenizer, os.path.join(MODEL_DIR, "tokenizer.pkl"))

    input_sequences = tokenizer.texts_to_sequences(inputs)
    output_sequences = tokenizer.texts_to_sequences(outputs)

    input_padded = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    output_padded = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return input_padded, output_padded, tokenizer

def split_and_scale(inputs, outputs):
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    return X_train, X_test, y_train, y_test

# Create Model
def create_model(input_shape):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(input_shape, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Train Model
def train_model(model, X_train, y_train):
    print("Training model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model, history

# Save Model
def save_model(model):
    print(f"Saving model to {MODEL_DIR}...")
    model.save(os.path.join(MODEL_DIR, "model.h5"))

# Main Program
if __name__ == "__main__":
    print("Loading TFRecords...")
    inputs, outputs = load_tfrecord_dataset(TFRECORD_PATTERN)

    print("Tokenizing and preprocessing data...")
    inputs, outputs, tokenizer = tokenize_and_preprocess(inputs, outputs)

    print("Processing and splitting data...")
    X_train, X_test, y_train, y_test = split_and_scale(inputs, outputs)

    print("Creating and training the model...")
    model = create_model(input_shape=X_train.shape[1])
    model, history = train_model(model, X_train, y_train)

    print("Saving the model and preprocessing tools...")
    save_model(model)

    print("Model training and saving complete.")
