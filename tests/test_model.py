from src.model_testing import test_model

def test_model_output():
    sentence = "She go to market."
    corrected = test_model(sentence)
    assert corrected == "She goes to market.", "Correction failed!"

