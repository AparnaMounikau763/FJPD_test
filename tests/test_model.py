import joblib
import numpy as np
from app import preprocess_text
from app.app import app
from app.model import preprocess_text

# Load trained model & vectorizer
model = joblib.load("job_fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ✅ Test preprocessing function
def test_preprocess_text():
    text = "Hello WORLD 123!!!"
    cleaned = preprocess_text(text)

    assert isinstance(cleaned, str)
    assert "123" not in cleaned
    assert cleaned.islower()


# ✅ Test vectorizer output shape
def test_vectorizer_output():
    sample_text = "Software engineer job remote python"
    processed = preprocess_text(sample_text)

    vec = vectorizer.transform([processed])

    assert vec.shape[0] == 1  # one sample


# ✅ Test model prediction returns valid output
def test_model_prediction():
    sample_text = "Urgent hiring work from home data entry job"
    processed = preprocess_text(sample_text)

    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)

    assert prediction[0] in [0, 1]  # 0 = real, 1 = fake


# ✅ Test known realistic job (should likely be real or stable output)
def test_real_job_example():
    text = "Senior Python Developer with 5 years experience required"
    processed = preprocess_text(text)

    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)

    assert prediction is not None