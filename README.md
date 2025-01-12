# Grammar Correction with T5 Transformer

This project fine-tunes a pre-trained T5 transformer model (`google/t5-base`) to perform grammar correction tasks. The dataset is provided in TFRecord format and includes sentences with grammatical errors along with their corrected counterparts.

---

## 📁 Project Structure

```plaintext
grammar_correction_project/
│
├── data/
│   └── grammar_dataset.tfrecord   # Dataset in TFRecord format
│
├── models/
│   └── fine_tuned_t5_base/        # Saved fine-tuned model
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── model_training.py          # Model fine-tuning logic
│   ├── model_testing.py           # Model testing logic
│   └── utils.py                   # Helper functions
│
├── tests/
│   └── test_model.py              # Unit tests for model and components
│
├── notebooks/
│   └── exploration.ipynb          # Optional dataset exploration
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── main.py                        # Entry point for training and testing
```

---

## 🚀 Features

- Fine-tunes the **T5 transformer** for grammar correction.
- Supports **TFRecord dataset format**.
- Modular design with Python standards (PEP8 compliant).
- Includes unit tests for reliability.
- Easily extendable for cross-lingual or advanced NLP tasks.

---

## 🔧 Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/grammar_correction_project.git
   cd grammar_correction_project
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**:
   - Place your TFRecord dataset in the `data/` directory.
   - Ensure the dataset follows the structure:
     - `input_text`: Sentences with grammatical errors.
     - `target_text`: Corrected sentences.

---

## 🏋️‍♂️ Fine-Tuning the Model

Run the `main.py` file to start fine-tuning:
```bash
python main.py
```
- The fine-tuned model will be saved in the `models/fine_tuned_t5_base/` directory.

---

## 🧪 Testing the Model

You can test the fine-tuned model with any input sentence:
```python
from src.model_testing import test_model

sentence = "She go to market."
corrected_sentence = test_model(sentence, model_path="models/fine_tuned_t5_base")
print(f"Corrected Sentence: {corrected_sentence}")
```

Expected Output:
```plaintext
Original: She go to market.
Corrected: She goes to market.
```

---

## 🧩 File Descriptions

- **`data_processing.py`**: Loads and tokenizes the dataset.
- **`model_training.py`**: Contains the logic for fine-tuning the T5 model.
- **`model_testing.py`**: Handles inference and testing of the fine-tuned model.
- **`test_model.py`**: Unit tests to validate the model.
- **`main.py`**: Entry point for fine-tuning and testing.
- **`requirements.txt`**: Lists all dependencies for the project.

---

## 📝 Future Enhancements

- **Cross-lingual Grammar Correction**: Support multiple languages using multilingual models like `mT5`.
- **Web Application**: Build a web-based interface using Flask or FastAPI for real-time corrections.
- **Batch Inference**: Add support for file uploads to process multiple sentences simultaneously.

---

## 🤝 Contribution

Feel free to fork this repository and submit pull requests for any improvements or additional features. Please ensure your contributions align with Python standards and include appropriate documentation.

---

## 📜 License

This project is licensed under the Student LICENSE.

---

## 📞 Contact

For any queries or assistance, feel free to reach out:
- **Name**: Anshuman Gaonsindhe
- **Email**: anshuman.gaonsindhe@gmail.com
- **GitHub**: [https://github.com/agaonsindhe](https://github.com/agaonsindhe)
