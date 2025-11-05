import pytest
import json
from app import app # Import the Flask 'app' object from your app.py file

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the root/health-check endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.data == b"Flask API is running!"

def test_prediction_survived(client):
    """Test the /predict endpoint with a known survivor."""
    # This is a known passenger who should survive
    # We use the lowercase column names from your titanic.csv
    test_data = {
        "pclass": "1",
        "sex": "female",
        "age": 38,
        "fare": 71.2833,
        "embarked": "C"
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    json_data = response.get_json()

    assert response.status_code == 200
    assert 'prediction' in json_data
    assert json_data['prediction'] == 'Survived'

def test_prediction_invalid(client):
    """Test the /predict endpoint with missing data."""
    # We send data with missing 'age', 'fare', 'embarked'
    test_data = {"pclass": "1", "sex": "female"} 
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    
    # It should fail gracefully with a 400 Bad Request
    assert response.status_code == 400 
    assert 'error' in response.get_json()