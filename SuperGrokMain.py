import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import re
import os
import logging
import requests
import speech_recognition as sr
import pyttsx3
import sentencepiece as spm
from flask import Flask, request, jsonify
from waitress import serve
from flask_talisman import Talisman

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model Definitions
class SuperGrok(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = hk.Embed(vocab_size=config['vocab_size'], embed_dim=config['d_model'])
        self.transformer = hk.nets.TransformerDecoder(
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            key_size=config['d_model'] // config['num_heads'],
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
        )
        self.output_layer = hk.Linear(config['vocab_size'])

    def __call__(self, x):
        x = self.embed(x)
        x = self.transformer(x, x)
        return self.output_layer(x)

# Configuration
config = {
    'vocab_size': 32000,
    'd_model': 16384,
    'num_layers': 128,
    'num_heads': 64,
    'max_length': 8192,
}

# Model Functions
def model_fn(x):
    module = SuperGrok(config)
    return module(x)

def init_model(rng):
    dummy_input = jnp.ones((1, config['max_length']), dtype=jnp.int32)
    init_fn = hk.without_apply_rng(hk.transform(model_fn))
    params = init_fn.init(rng, dummy_input)
    return params

def apply_model(params, input_ids):
    apply_fn = hk.without_apply_rng(hk.transform(model_fn)).apply
    return apply_fn(params, input_ids)

# Load Model
def load_model():
    try:
        with open('model_checkpoint.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.warning("No model checkpoint found, initializing new model.")
        return init_model(jax.random.PRNGKey(42))

params = load_model()

# Tokenization
sp = spm.SentencePieceProcessor()
sp.Load('supergrok.model')

# Context Retention (Local Memory Cache)
conversation_history = []

def sanitize_input(text):
    return re.sub(r'<[^>]*>', '', text)[:1000]

def tokenize(text):
    return sp.encode_as_ids(text)

def prepare_for_model(text, max_length=config['max_length']):
    tokenized = tokenize(text)
    if len(tokenized) < max_length:
        tokenized += [0] * (max_length - len(tokenized))
    else:
        tokenized = tokenized[:max_length]
    return jnp.array(tokenized)

# Flask API Server
app = Flask(__name__)
talisman = Talisman(app, content_security_policy={'default-src': "'self'"})

@app.route('/predict', methods=['POST'])
def predict():
    global conversation_history
    try:
        input_data = request.json.get('input', '')
        sanitized_input = sanitize_input(input_data)
        conversation_history.append(sanitized_input)
        
        if len(conversation_history) > 10:
            conversation_history.pop(0)  # Keep memory manageable

        input_ids = prepare_for_model(" ".join(conversation_history))
        predictions = apply_model(params, input_ids[None, :])
        decoded_prediction = sp.decode([jnp.argmax(predictions[0])])

        conversation_history.append(decoded_prediction)
        return jsonify({'prediction': decoded_prediction}), 200
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

# File Upload Handling
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    
    return jsonify({'status': f'File {file.filename} uploaded successfully', 'file_path': file_path}), 200

# Voice Input/Output
engine = pyttsx3.init()
recognizer = sr.Recognizer()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Error with speech recognition service."

# Start API and Interactive Chat
if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080, threads=4)
    
    print("
Super GROK is running! You can now chat directly from the terminal.")
    print("Type your message and press Enter. Type 'exit' or 'quit' to stop.
")

    while True:
        user_input = input("Ask Super GROK (or say 'voice' for speech mode): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "voice":
            user_input = listen()
            print(f"You said: {user_input}")
        
        response = requests.post("http://localhost:8080/predict", json={"input": user_input})
        
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            print("Super GROK:", prediction)
            speak(prediction)  # AI will also respond with voice
        else:
            print("Error:", response.json().get("error"))
