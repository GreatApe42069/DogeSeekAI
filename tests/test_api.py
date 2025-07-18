import pytest
import requests
import time
from fastapi.testclient import TestClient
from DogeSeekAIMain import app  # Adjust to your app file's path

client = TestClient(app)

def test_predict_success():
    """Test /api/predict with valid input."""
    response = client.post(
        "/api/predict",
        json={"question": "What is Dogecoin?", "context": "Dogecoin is a cryptocurrency created in 2013."}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["answer"] == "a cryptocurrency"

def test_predict_no_context():
    """Test /api/predict without context (should use default for Dogecoin)."""
    response = client.post(
        "/api/predict",
        json={"question": "What is Dogecoin?", "context": ""}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "cryptocurrency" in response.json()["answer"]

def test_predict_invalid_input():
    """Test /api/predict with missing question."""
    response = client.post(
        "/api/predict",
        json={"context": "Dogecoin is a cryptocurrency."}
    )
    assert response.status_code == 422  # FastAPI validation error

def test_voice_success():
    """Test /api/voice with a sample audio file."""
    # Create a small dummy WAV file (requires actual file for real test)
    with open("tests/sample_audio.wav", "rb") as f:
        response = client.post("/api/voice", files={"file": f})
    assert response.status_code == 200
    assert "transcription" in response.json()
    assert "emotion" in response.json()

def test_voice_empty_file():
    """Test /api/voice with an empty file."""
    with open("tests/empty.wav", "wb") as f:
        f.write(b"")
    with open("tests/empty.wav", "rb") as f:
        response = client.post("/api/voice", files={"file": f})
    assert response.status_code == 500  # Assuming error for invalid audio

def test_image_success():
    """Test /api/image with a sample image file."""
    # Create a small dummy JPG file (requires actual file for real test)
    with open("tests/sample_image.jpg", "rb") as f:
        response = client.post("/api/image", files={"file": f})
    assert response.status_code == 200
    assert "features" in response.json()

def test_image_empty_file():
    """Test /api/image with an empty file."""
    with open("tests/empty.jpg", "wb") as f:
        f.write(b"")
    with open("tests/empty.jpg", "rb") as f:
        response = client.post("/api/image", files={"file": f})
    assert response.status_code == 500  # Assuming error for invalid image

@pytest.mark.integration
def test_train_federated():
    """Test /api/train_federated endpoint (requires Flower server running)."""
    response = client.post("/api/train_federated")
    assert response.status_code == 200
    assert response.json() == {"status": "Federated training started"}

@pytest.mark.performance
def test_predict_performance():
    """Test /api/predict response time with large context."""
    large_context = "Dogecoin is a cryptocurrency created in 2013. " * 1000
    start_time = time.time()
    response = client.post(
        "/api/predict",
        json={"question": "What is Dogecoin?", "context": large_context}
    )
    elapsed_time = time.time() - start_time
    assert response.status_code == 200
    assert elapsed_time < 5  # Adjust threshold (e.g., 5 seconds)