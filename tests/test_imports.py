import traceback
try:
    print("Testing torch...")
    import torch
    print("Torch loaded!")
    print("Testing transformers...")
    from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC
    print("Transformers loaded!")
    print("Testing torchvision...")
    import torchvision.models
    print("Torchvision loaded!")
    print("Testing ipfshttpclient...")
    import ipfshttpclient
    print("IPFS loaded!")
    print("Testing DistilBert...")
    text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    text_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    print("DistilBert loaded!")
    print("Testing Wav2Vec2...")
    speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    print("Wav2Vec2 loaded!")
    print("Testing ResNet50...")
    resnet_model = torchvision.models.resnet50(pretrained=True)
    resnet_model.eval()
    print("ResNet50 loaded!")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()