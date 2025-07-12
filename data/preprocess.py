from datasets import load_dataset
import json
import ipfshttpclient
from doginals.inscribe import DoginalInscriber
from cryptography.fernet import Fernet

def preprocess_and_store(config, use_doginal=False):
    ipfs_client = ipfshttpclient.connect(config["ipfs_node"])
    doge_client = DoginalInscriber(config["doge_address"], config["doge_rpc"]) if config.get("doge_address") and config.get("doge_rpc") else None
    cipher = Fernet(Fernet.generate_key())
    
    text_data = load_dataset("wikipedia", "20220301.en", split="train[:1000]")
    text_processed = [{"text": item["text"].replace("<[^>]+>", "")} for item in text_data]
    with open("data/text_data.json", "w") as f:
        json.dump(text_processed, f)
    
    encrypted_data = cipher.encrypt(open("data/text_data.json", "rb").read())
    ipfs_hash = ipfs_client.add_bytes(encrypted_data)["Hash"]
    storage_info = {"text_data_ipfs": ipfs_hash}
    
    if use_doginal and doge_client:
        try:
            txid = doge_client.inscribe(encrypted_data, config["parent_doginal_id"])
            storage_info["text_data_txid"] = txid
        except Exception as e:
            storage_info["error"] = f"Doginal inscription failed: {str(e)}"
    
    with open("data/cids.json", "a") as f:
        json.dump(storage_info, f)
        f.write("\n")
    
    return storage_info

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    preprocess_and_store(config)
