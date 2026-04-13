import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ✅ Test Home Page Loads
def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"html" in response.data.lower()


# ✅ Test Predict Route (POST request)
def test_predict_route(client):
    response = client.post('/predict', data={
        "title": "Software Engineer",
        "company_profile": "We are a tech company",
        "description": "Looking for Python developer",
        "location": "India"
    })

    assert response.status_code == 200
    assert b"Job Posting" in response.data  # Fake or Real result appears


# ✅ Test Missing Fields Handling (Edge Case)
def test_predict_missing_fields(client):
    response = client.post('/predict', data={
        "title": "",
        "company_profile": "",
        "description": "",
        "location": ""
    })

    assert response.status_code == 200