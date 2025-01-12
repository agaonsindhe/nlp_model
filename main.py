from src.model_training import fine_tune_t5
from src.model_testing import test_model

if __name__ == "__main__":
    # Fine-tune the model
    dataset_path = "data/grammar_dataset.tfrecord"
    output_dir = "models/fine_tuned_t5_base"
    fine_tune_t5(dataset_path, output_dir)

    # Test the model
    test_sentence = "She go to school every day."
    corrected_sentence = test_model(test_sentence, output_dir)
    print(f"Original: {test_sentence}")
    print(f"Corrected: {corrected_sentence}")
