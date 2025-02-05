import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pickle
import re
import os
import json
import sentencepiece as spm
from typing import List, Tuple, Dict, Any
from flask import Flask, request, jsonify
from waitress import serve
from flask_talisman import Talisman
from jax.experimental import optimizers

# Security & Logging
import logging
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# 🔒 SECURITY: Encryption Key (Generated & Stored Securely)
# -------------------------------
ENCRYPTION_KEY = b"YOUR_256_BIT_KEY_HERE"  # Replace with a securely generated key
cipher = Fernet(ENCRYPTION_KEY)

# -------------------------------
# ⚙️ MODEL CONFIGURATION
# -------------------------------
config = {
    "vocab_size": 32000,  # Match SentencePiece model
    "d_model": 16384,
    "num_layers": 128,
    "num_heads": 64,
    "max_length": 8192,
    "learning_rate": 1e-4,  # Learning rate for fine-tuning
}

# -------------------------------
# 🔥 TRANSFORMER MODEL
# -------------------------------
class SuperGrok(hk.Module):
    def __init__(self, config: Dict[str, int]):
        super().__init__()
        self.config = config
        self.embed = hk.Embed(vocab_size=config["vocab_size"], embed_dim=config["d_model"])
        self.transformer = hk.nets.TransformerDecoder(
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            key_size=config["d_model"] // config["num_heads"],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        self.output_layer = hk.Linear(config["vocab_size"])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.embed(x)
        x = self.transformer(x, x)
        return self.output_layer(x)

def model_fn(x: jnp.ndarray) -> jnp.ndarray:
    module = SuperGrok(config)
    return module(x)

init_fn, apply_fn = hk.without_apply_rng(hk.transform(model_fn))

# -------------------------------
# 🔄 MODEL LOADING & SAVING
# -------------------------------
def save_model(params: hk.Params) -> None:
    encrypted_params = cipher.encrypt(pickle.dumps(params))
    with open("model_checkpoint.enc", "wb") as f:
        f.write(encrypted_params)

def load_model() -> hk.Params:
    if os.path.exists("model_checkpoint.enc"):
        with open("model_checkpoint.enc", "rb") as f:
            decrypted_data = cipher.decrypt(f.read())
            return pickle.loads(decrypted_data)
    logging.warning("No model checkpoint found, initializing new model.")
    return init_fn.init(jax.random.PRNGKey(42), jnp.ones((1, config["max_length"]), dtype=jnp.int32))

params = load_model()

# -------------------------------
# 📖 TOKENIZATION
# -------------------------------
sp = spm.SentencePieceProcessor()
sp.Load("supergrok.model")

def sanitize_input(text: str) -> str:
    return re.sub(r"<[^>]*>", "", text)[:1000]  # Remove HTML tags, limit length

def tokenize(text: str) -> List[int]:
    return sp.encode_as_ids(text)

def prepare_for_model(text: str) -> jnp.ndarray:
    tokenized = tokenize(text)
    if len(tokenized) < config["max_length"]:
        tokenized += [0] * (config["max_length"] - len(tokenized))
    else:
        tokenized = tokenized[:config["max_length"]]
    return jnp.array(tokenized)

# -------------------------------
# 🔥 REAL-TIME LEARNING (Fine-Tuning)
# -------------------------------
optimizer = optax.adam(config["learning_rate"])
opt_state = optimizer.init(params)

def fine_tune(params: hk.Params, opt_state: Any, input_ids: jnp.ndarray) -> Tuple[hk.Params, Any]:
    def loss_fn(p):
        predictions = apply_fn(p, input_ids[None, :])  # Add batch dim
        target = input_ids[1:] + [0]  # Shifted target
        return jnp.mean((predictions - jnp.array(target)) ** 2)

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state

# -------------------------------
# 🤖 FLASK API SERVER
# -------------------------------
app = Flask(__name__)
Talisman(app, content_security_policy={"default-src": "'self'"})

# 🔵 Context Memory Cache (Local)
context_memory = []

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json.get("input", "")
        sanitized_input = sanitize_input(input_data)
        
        # 📌 Cache recent inputs
        context_memory.append(sanitized_input)
        if len(context_memory) > 10:  # Limit cache size
            context_memory.pop(0)

        input_ids = prepare_for_model(sanitized_input)
        
        # 🏎️ XLA Compilation for Speed
        fast_apply = jax.jit(apply_fn)
        predictions = fast_apply(params, input_ids[None, :])  

        # 🌟 Top-K Sampling for better responses
        top_k = 5
        sorted_indices = jnp.argsort(predictions[0])[-top_k:]
        decoded_prediction = sp.decode([int(sorted_indices[jnp.argmax(jnp.take(predictions[0], sorted_indices))])])

        # 🏋️ Real-time fine-tuning
        global params, opt_state
        params, opt_state = fine_tune(params, opt_state, input_ids)
        save_model(params)

        return jsonify({"prediction": decoded_prediction}), 200
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/update_model", methods=["POST"])
def update_model():
    try:
        new_params = load_model()
        save_model(new_params)
        return jsonify({"status": "Model updated successfully"}), 200
    except Exception as e:
        logging.error(f"Error updating model: {e}")
        return jsonify({"error": "Model update failed"}), 500

# -------------------------------
# 🚀 RUN SERVER
# -------------------------------
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
