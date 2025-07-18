import traceback
try:
    print("Starting WoW Much Imports...")
    import json
    import torch
    import librosa
    import numpy as np
    import pyttsx3
    import requests
    from fastapi import FastAPI, File, UploadFile, HTTPException, Body
    from fastapi.middleware.cors import CORSMiddleware
    from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, Wav2Vec2ForCTC, Wav2Vec2Processor
    from PIL import Image
    import torchvision.transforms as transforms
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    import ipfshttpclient
    from cryptography.fernet import Fernet
    import flwr
    import os
    from doginals.inscribe import DoginalInscriber
    from federated.federated_client import DogeSeekAIClient
    print("Such Imports Very Complete LFG!!!")

    # Load configuration
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # FastAPI setup
    print("Initializing FastAPI...")
    app = FastAPI(title="DogeSeekAI API", description="DogeSeekAI Decentralized Multimodal AI")
    print("Much FastAPI So Initialized!")

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load models
    print("Loading Much Models, Very Wow...")
    text_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased-distilled-squad")
    text_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet_model.eval()
    tts_engine = pyttsx3.init()
    print("Models loaded, such AI!")

    print("Much Connecting to IPFS So WoW...")
    # IPFS client
    try:
        ipfs_client = ipfshttpclient.connect(config["ipfs_node"])
        print("So Connected Such IPFS")
    except Exception as e:
        print(f"IPFS connection failed: {str(e)}")
        ipfs_client = None

    # Dogecoin client
    doge_client = None
    wallet_path = "doginals/.wallet.json"
    doge_address = config.get("doge_address", "")
    private_key = ""
    if os.path.exists(wallet_path):
        with open(wallet_path, "r") as wallet_file:
            wallet_data = json.load(wallet_file)
            doge_address = wallet_data.get("address", "")
            private_key = wallet_data.get("private_key", "")
    if doge_address and config.get("doge_rpc"):
        try:
            doge_client = DoginalInscriber(doge_address, config["doge_rpc"], private_key=private_key)
        except Exception as e:
            print(f"Dogecoin client initialization failed: {str(e)}")

    @app.get("/test")
    async def test():
        return {"message": "Server is running"}

    # Encryption
    key = Fernet.generate_key()
    cipher = Fernet(key)

    # Conversation history
    conversation_history = []

    # FastAPI endpoints
    @app.post("/api/predict", summary="Text-based Q&A")
    async def predict(question: str = Body(...), context: str = Body(default="")):
        global conversation_history
        if config["external_apis"]["generic_llm"]["api_key"] and config["external_apis"]["generic_llm"]["endpoint"]:
            try:
                response = requests.post(
                    config["external_apis"]["generic_llm"]["endpoint"],
                    headers={"Authorization": f"Bearer {config['external_apis']['generic_llm']['api_key']}"},
                    json={"prompt": f"Question: {question}\nContext: {context}"}
                )
                answer = response.json().get("answer", "External API failed")
            except Exception as e:
                answer = f"External API error: {str(e)}"
        else:
            try:
                inputs = text_tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                with torch.no_grad():
                    outputs = text_model(**inputs)
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1
                print(f"Debug: answer_start={answer_start}, answer_end={answer_end}, input_ids_shape={inputs['input_ids'].shape}")
                if 0 < answer_start <= answer_end <= inputs["input_ids"].shape[1]:
                    answer = text_tokenizer.decode(
                        inputs["input_ids"][0][answer_start:answer_end],
                        skip_special_tokens=True
                    )
                    if not answer or answer.lower() == question.lower():  # Fallback if model fails
                        answer = context if context else "I don't have enough information to answer. Please provide context or try a different question."
                else:
                    answer = context if context else "I don't have enough information to answer. Please provide context or try a different question."
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                answer = context if context else "Prediction failed due to an internal error."
        
        conversation_history.append({"question": question, "answer": answer})
        if len(conversation_history) > config["context_window"]:
            conversation_history.pop(0)
        
        try:
            tts_engine.say(answer)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS engine failed: {str(e)}")
        
        return {"answer": answer}

    @app.post("/api/voice", summary="Speech-to-text and emotion analysis")
    async def voice(file: UploadFile = File(...)):
        try:
            file_path = "temp_audio.wav"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            y, sr = librosa.load(file_path)
            inputs = speech_processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
            logits = speech_model(inputs.input_values).logits
            transcription = speech_processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            
            emotion = "Neutral"  # Placeholder, as emotion analysis is not implemented
            os.remove(file_path)
            
            return {"transcription": transcription, "emotion": emotion}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

    @app.post("/api/image", summary="Image feature extraction")
    async def image(file: UploadFile = File(...)):
        try:
            file_path = "temp_image.jpg"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            image = Image.open(file_path).convert("RGB")
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            image_tensor = transform(image).unsqueeze(0)
            features = resnet_model(image_tensor).detach().numpy().tolist()
            os.remove(file_path)
            
            return {"features": features}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

    @app.post("/api/upload_to_storage", summary="Store data on IPFS or Dogecoin Blockchain")
    async def upload_to_storage(file: UploadFile = File(...), use_doginal: bool = False):
        try:
            content = await file.read()
            encrypted_content = cipher.encrypt(content)
            
            if ipfs_client:
                ipfs_hash = ipfs_client.add_bytes(encrypted_content)["Hash"]
            else:
                raise Exception("IPFS client not initialized")
            
            storage_info = {"ipfs_hash": ipfs_hash}
            
            if use_doginal and doge_client:
                try:
                    txid = doge_client.inscribe(encrypted_content, config["parent_doginal_id"])
                    storage_info["txid"] = txid
                except Exception as e:
                    storage_info["error"] = f"Doginal inscription failed: {str(e)}"
            
            with open("data/cids.json", "a") as f:
                json.dump(storage_info, f)
                f.write("\n")
            
            return storage_info
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Storage upload failed: {str(e)}")

    @app.post("/api/train_federated", summary="Start federated training")
    async def train_federated():
        try:
            client = DogeSeekAIClient(text_model, text_tokenizer)
            flwr.client.start_client(server_address=config["federated_server"], client=client.to_client())
            return {"status": "Federated training started"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Federated training failed: {str(e)}")

except Exception as e:
    print(f"Error in DogeSeekAIMain.py: {e}")
    traceback.print_exc()

# Only run the server if app is defined
if app is not None and __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=config["port"], log_level="debug")
    except Exception as e:
        print(f"Uvicorn server failed: {str(e)}")
    finally:
        try:
            tts_engine.stop()  # Explicitly stop pyttsx3 engine
        except NameError:
            print("TTS engine not initialized, skipping stop.")
else:
    print("Initialization failed, server not started.")