import pytest
import requests
from fastapi.testclient import TestClient
from api.DogeSeekAIMain import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/api/predict", json={"question": "What is Dogecoin?", "context": "Dogecoin is a cryptocurrency."})
    assert response.status_code == 200
    assert "answer" in response.json()

def test_voice_endpoint():
    with open("tests/sample_audio.wav", "wb") as f:
        f.write(b"")
    with open("tests/sample_audio.wav", "rb") as f:
        response = client.post("/api/voice", files={"file": f})
    assert response.status_code == 200
    assert "transcription" in response.json()

def test_image_endpoint():
    with open("tests/sample_image.jpg", "wb") as f:
        f.write(b"")
    with open("tests/sample_image.jpg", "rb") as f:
        response = client.post("/api/image", files={"file": f})
    assert response.status_code == 200
    assert "features" in response.json()
