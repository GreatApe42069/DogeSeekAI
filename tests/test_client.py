import pytest
import requests
import json
import time
import os
from client.client import DogeSeekAIClient
from unittest.mock import patch

@pytest.fixture
def client():
    """Fixture to initialize DogeSeekAIClient."""
    return DogeSeekAIClient()

# ---------------- API Interaction Tests (Real API Calls) ----------------
def test_predict_real_api(client):
    """Test real API call to /api/predict."""
    response = client.run_predict("What is Dogecoin?", "Dogecoin is a cryptocurrency.")
    assert response["answer"] == "a cryptocurrency"

def test_predict_no_context(client):
    """Test real API call to /api/predict without context."""
    response = client.run_predict("What is Dogecoin?", "")
    assert "cryptocurrency" in response["answer"]

def test_voice_real_api(client):
    """Test real API call to /api/voice with a sample file."""
    with open("tests/sample_audio.wav", "rb") as f:
        response = client.run_voice(f)  # Adjust to your client method
        assert response.status_code == 200
        assert "transcription" in response.json()

# ---------------- Dogecoin Tests (Real Connections) ----------------
@pytest.mark.integration
def test_dogecoin_connection(client):
    """Test real connection to Dogecoin node."""
    try:
        response = client.inscriber.get_node_info()  # Adjust to your method
        assert response is not None
    except Exception as e:
        pytest.fail(f"Dogecoin connection failed: {str(e)}")

@pytest.mark.integration
def test_dogecoin_version(client):
    """Test retrieving Dogecoin Core version."""
    try:
        version = client.inscriber.get_version()  # Adjust to your method
        assert version.startswith("1.")  # Adjust based on expected version
    except Exception as e:
        pytest.fail(f"Dogecoin version check failed: {str(e)}")

# ---------------- Dogecoin Tests (Mocked for Sensitive Operations) ----------------
def test_dogecoin_transaction_broadcast(client):
    """Test transaction broadcast with mock."""
    with patch('doginals.inscribe.DoginalInscriber.inscribe') as mock_inscribe:
        mock_inscribe.return_value = "fake_tx_id"
        tx_id = client.inscriber.inscribe("test_data", "")  # Adjust to your method
        assert tx_id == "fake_tx_id"

def test_dogecoin_wallet_connection(client):
    """Test wallet connection with mock."""
    with patch('doginals.inscribe.DoginalInscriber.get_address') as mock_get_address:
        mock_get_address.return_value = "fake_address"
        address = client.inscriber.get_address()  # Adjust to your method
        assert address == "fake_address"

# ---------------- IPFS Tests (Real Connections) ----------------
@pytest.mark.integration
def test_ipfs_connection(client):
    """Test real connection to IPFS daemon."""
    try:
        ipfs_client = client.ipfs_client  # Adjust to your client attribute
        assert ipfs_client is not None
    except Exception as e:
        pytest.fail(f"IPFS connection failed: {str(e)}")

@pytest.mark.integration
def test_ipfs_version(client):
    """Test retrieving IPFS version."""
    try:
        version = client.ipfs_client.version()["Version"]  # Adjust to your method
        assert version.startswith("0.")  # Adjust based on your IPFS version (0.35.0)
    except Exception as e:
        pytest.fail(f"IPFS version check failed: {str(e)}")

@pytest.mark.integration
def test_ipfs_add_file(client):
    """Test adding a small file to IPFS."""
    try:
        hash = client.ipfs_client.add_bytes(b"Test data")["Hash"]
        assert hash is not None
    except Exception as e:
        pytest.fail(f"IPFS file add failed: {str(e)}")

# ---------------- Federated Learning Tests ----------------
@pytest.mark.integration
def test_train_federated_start(client):
    """Test /api/train_federated endpoint (requires Flower server running)."""
    response = requests.post("http://localhost:8080/api/train_federated")
    assert response.status_code == 200
    assert response.json() == {"status": "Federated training started"}

# ---------------- Configuration Tests ----------------
def test_load_config_valid(tmp_path):
    """Test loading a valid config.json."""
    config_data = {
        "port": 8080,
        "federated_server": "localhost:8083",
        "ipfs_node": "/ip4/127.0.0.1/tcp/5001",
        "doge_rpc": "http://127.0.0.1:22555"
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    client = DogeSeekAIClient(config_path=str(config_path))  # Adjust constructor
    assert client.base_url == "http://localhost:8080"

def test_load_config_invalid(tmp_path):
    """Test handling an invalid config.json."""
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        f.write("invalid json")
    with pytest.raises(json.JSONDecodeError):
        DogeSeekAIClient(config_path=str(config_path))

# ---------------- Security Tests ----------------
def test_encryption_decryption(client):
    """Test encryption and decryption of sensitive data."""
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    cipher = Fernet(key)
    original_data = b"sensitive_key"
    encrypted = cipher.encrypt(original_data)
    decrypted = cipher.decrypt(encrypted)
    assert decrypted == original_data

def test_unauthorized_api_request():
    """Test that API rejects unauthorized requests (if auth implemented)."""
    response = requests.post("http://localhost:8080/api/predict", json={})
    assert response.status_code in [422, 401]  # 422 for validation, 401 if auth added

# ---------------- Performance Tests ----------------
@pytest.mark.performance
def test_api_response_time_large_file(client):
    """Test API response time with a large audio file."""
    # Create a large dummy WAV file (requires actual file for real test)
    with open("tests/large_test.wav", "rb") as f:
        start_time = time.time()
        response = client.run_voice(f)  # Adjust to your method
        elapsed_time = time.time() - start_time
    assert response.status_code == 200
    assert elapsed_time < 10  # Adjust threshold