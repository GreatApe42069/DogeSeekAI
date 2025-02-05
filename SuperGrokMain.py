import jax
import jax.numpy as jnp
import haiku as hk
import optax
import flask
from flask import Flask, request, jsonify
from flask_talisman import Talisman
import sentencepiece as spm
import json
import logging
import os
import speech_recognition as sr
import pyttsx3
from cryptography.fernet import Fernet

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load configuration from file
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = {
        "d_model": 16384,
        "num_layers": 128,
        "max_length": 1024,
        "context_window": 10,
        "encryption_key": Fernet.generate_key().decode()
    }

# Encryption setup
fernet = Fernet(config["encryption_key"].encode())

# AI Model
class SuperGrok(hk.Module):
    def __call__(self, x):
        return hk.Linear(config["d_model"])(x)

def model_fn(x):
    net = SuperGrok()
    return net(x)

# Initialize model
model = hk.transform(model_fn)
params = model.init(jax.random.PRNGKey(42), jnp.ones([1, config["d_model"]]))

# Flask App Setup
app = Flask(__name__)
Talisman(app)

# Context retention
conversation_history = []

# Voice Engine
engine = pyttsx3.init()
recognizer = sr.Recognizer()

def sanitize_input(user_input):
    return user_input.replace("<", "").replace(">", "")

@app.route("/predict", methods=["POST"])
def predict():
    global conversation_history
    data = request.get_json()
    user_input = sanitize_input(data.get("input", ""))

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Update conversation history
    conversation_history.append(user_input)
    if len(conversation_history) > config["context_window"]:
        conversation_history.pop(0)

    response_text = f"Super GROK response to: {user_input}"

    # Encrypt and save conversation
    encrypted_data = fernet.encrypt(response_text.encode())
    with open("model_checkpoint.enc", "wb") as f:
        f.write(encrypted_data)

    return jsonify({"response": response_text})

@app.route("/voice", methods=["GET"])
def voice_input():
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return jsonify({"voice_input": text})
        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"}), 400

@app.route("/update_model", methods=["POST"])
def update_model():
    global params
    if os.path.exists("model_checkpoint.enc"):
        with open("model_checkpoint.enc", "rb") as f:
            encrypted_data = f.read()
            decrypted_data = fernet.decrypt(encrypted_data).decode()
            logging.info("Model Updated: %s", decrypted_data)
    return jsonify({"message": "Model updated successfully!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
