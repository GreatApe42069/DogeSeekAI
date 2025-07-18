import pytest
import traceback
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, Wav2Vec2Processor, Wav2Vec2ForCTC
import torchvision.models
import ipfshttpclient
from torchvision.models import ResNet50_Weights
import numpy as np

def test_torch_import():
    """Test importing torch."""
    print("Testing torch...")
    try:
        import torch
        print("Torch loaded!")
        assert torch.__version__ is not None
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"Torch import failed: {str(e)}")

def test_transformers_import():
    """Test importing transformers models."""
    print("Testing transformers...")
    try:
        from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, Wav2Vec2Processor, Wav2Vec2ForCTC
        print("Transformers loaded!")
        assert True
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"Transformers import failed: {str(e)}")

def test_torchvision_import():
    """Test importing torchvision models."""
    print("Testing torchvision...")
    try:
        import torchvision.models
        print("Torchvision loaded!")
        assert True
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"Torchvision import failed: {str(e)}")

def test_ipfshttpclient_import():
    """Test importing ipfshttpclient."""
    print("Testing ipfshttpclient...")
    try:
        import ipfshttpclient
        print("IPFS loaded!")
        assert ipfshttpclient.__version__ is not None
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"IPFS import failed: {str(e)}")

def test_distilbert_loading():
    """Test loading DistilBERT model and tokenizer."""
    print("Testing DistilBert...")
    try:
        text_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased-distilled-squad")
        text_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
        print("DistilBert loaded!")
        assert text_tokenizer is not None
        assert text_model is not None
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"DistilBert loading failed: {str(e)}")

def test_distilbert_functionality():
    """Test DistilBERT processing with sample input."""
    print("Testing DistilBert functionality...")
    try:
        text_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased-distilled-squad")
        text_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
        inputs = text_tokenizer(
            "What is Dogecoin?",
            "Dogecoin is a cryptocurrency created in 2013.",
            return_tensors="pt"
        )
        outputs = text_model(**inputs)
        assert outputs.start_logits is not None
        assert outputs.end_logits is not None
        print("DistilBert functionality tested!")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"DistilBert functionality failed: {str(e)}")

def test_wav2vec2_loading():
    """Test loading Wav2Vec2 model and processor."""
    print("Testing Wav2Vec2...")
    try:
        speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        print("Wav2Vec2 loaded!")
        assert speech_processor is not None
        assert speech_model is not None
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"Wav2Vec2 loading failed: {str(e)}")

def test_wav2vec2_functionality():
    """Test Wav2Vec2 processing with sample input."""
    print("Testing Wav2Vec2 functionality...")
    try:
        speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        # Create a dummy audio input (1-second silence at 16kHz)
        audio_input = np.zeros(16000, dtype=np.float32)
        inputs = speech_processor(audio_input, sampling_rate=16000, return_tensors="pt")
        logits = speech_model(inputs.input_values).logits
        assert logits is not None
        print("Wav2Vec2 functionality tested!")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"Wav2Vec2 functionality failed: {str(e)}")

def test_resnet50_loading():
    """Test loading ResNet50 model."""
    print("Testing ResNet50...")
    try:
        resnet_model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet_model.eval()
        print("ResNet50 loaded!")
        assert resnet_model is not None
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"ResNet50 loading failed: {str(e)}")

def test_resnet50_functionality():
    """Test ResNet50 processing with sample input."""
    print("Testing ResNet50 functionality...")
    try:
        resnet_model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet_model.eval()
        # Create a dummy image input (224x224 RGB)
        dummy_image = torch.rand(1, 3, 224, 224)
        outputs = resnet_model(dummy_image)
        assert outputs.shape == (1, 1000)  # ResNet50 outputs 1000 classes
        print("ResNet50 functionality tested!")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pytest.fail(f"ResNet50 functionality failed: {str(e)}")