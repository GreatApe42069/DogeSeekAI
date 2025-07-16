```
        ██████╗  ██████╗  ██████╗ ███████╗███████╗███████╗███████╗██╗  ██╗ █████╗ ██╗
        ██╔══██╗██╔═══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝██╔════╝██║ ██╔╝██╔══██╗██║
        ██║  ██║██║   ██║██║  ███╗█████╗  █████╗  █████╗  █████╗  █████╔╝ ███████║██║
        ██║  ██║██║   ██║██║   ██║██╔══╝      ██║ ██╔══╝  ██╔══╝  ██╔═██╗ ██╔══██║██║
        ██████╔╝╚██████╔╝╚██████╔╝███████╗██████║ ███████╗███████╗██║  ██╗██║  ██║██║
        ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                  🐶 DogeSeekAI CLI - Much Wow, Such AI! 🐶
```
# 🐶DogeSeekAI: Much Wow, Such Blockchain!🚀

Welcome to the DogeSeekAI module! A Decentralized, End-to-End Encrypted, Open-source, Multimodal AI. It learns in real-time from user interactions and publicly available data, continuously improving itself while using GitHub and IPFS Desktop for modal data's large file storage, or while keeping all user data securely stored on your device. This module lets you inscribe doginals in the form of files, text, images, audio, or DRC-20 tokens onto the Dogecoin blockchain for immutable, decentralized storage. Bark loud, mint proud! Below are Doge-themed instructions and examples to get your Doge on the blockchain. 🌙

*Note: This model is currently still in development and testing stages.*

**Important Note**: Use a fresh Dogecoin wallet for minting to avoid conflicts until proper wallet support is available. Ensure your Dogecoin node is running with `rpcallowip=127.0.0.1` in `dogecoin.conf`. Verify that `doginals/.env` settings match `dogecoin.conf` for `rpc_user`, `rpc_password`, and `rpcport`.

---

## ✨ Features

- **Multimodal**: Text (DistilBERT), voice (Wav2Vec2), image (ResNet50).
- **Storage**: 
  - **GitHub**: Code, configs, small datasets (<100MB), metadata (free).
  - **IPFS**: Large datasets, model weights (free, requires IPFS installed and running).
  - **Doginals**: Optional immutable metadata/NFTs via `doginals.js` (costs DOGE, requires Node.js and Dogecoin Core). Enable Doginals storage in `config.json`.
- **Federated Learning**: Collaborative training via Flower.
- **E2EE**: AES-256 encryption for secure data storage.
- **Built-in Client**: Dogecoin-themed CLI for easy interaction.
- **External API**: Integrate into custom apps/websites.
- **API Key Support**: Supercharge with external model APIs.
- **Automation**: GitHub Actions for dataset preprocessing, model uploads, testing, and dependency updates.
- **Doginals Integration**: Inscribe files (images, audio, text), raw data, or DRC-20 tokens to the Dogecoin blockchain.

---

## 🛠️ Installation

### Prerequisites
- **Python**: Version 3.8 or later.
- **Node.js**: Version 20.17.0 (LTS) or later (v21.6.1 tested).
- **Dogecoin Core🐶**: Version 1.14.7 or later for blockchain interactions.
- **IPFS Desktop**: For decentralized storage of large datasets and model weights.
- **Git**: For cloning the repository.
- **Windows**: Tested on Windows 10/11; other OS may work with adjustments.
- **Fresh Dogecoin Wallet**: Recommended for minting to avoid conflicts.

### Dependencies
- **Python Dependencies** (`requirements.txt`):
  ```plaintext
  fastapi
  uvicorn
  ipfshttpclient
  requests
  python-dotenv
  pydub  # For audio processing
  Pillow  # For image processing
  ```

## Setup Steps

### Clone the Repository:
```bash
cd C:\Users\<YourUsername>\OneDrive\Desktop
git clone https://github.com/GreatApe42069/DogeSeekAI.git
cd DogeSeekAI
```

### Install Node.js and Doginals Dependencies: Managed in doginals/package.json. 
*After cloning repo Install with:*
```bash
cd doginals
npm install
npm audit fix
```

### Install Python 3.8+:Ubuntu/Debian:
```bash
cd DogeSeekAI
sudo apt update && sudo apt install python3.8 python3.8-venv python3.8-dev
```

or
### macOS (using Homebrew):
```bash
cd DogeSeekAI
brew install python@3.8
```

or
### Windows:Using Chocolatey (recommended):
```bash
cd DogeSeekAI
choco install python38
```

#### Install Chocolatey first for windows: 
https://chocolatey.org/install

***Or download from:*** 

https://www.python.org/downloads/release/python-3810/ (check "Add Python to PATH").

#### Verify:
```bash
python --version
```
#### Install Python Dependencies:
```bash
pip install -r requirements.txt
```

### Install Node.js:
Download from https://nodejs.org/en/download (LTS recommended).

#### Verify:
```bash
node --version
npm --version
```

*Expected Output: v20.17.0 or later, npm@10.9.1 or later.*

### 🐶Install Dogecoin Core:
Download from https://dogecoin.com/

#### Configure C:\Users\<YourUsername>\AppData\Roaming\Dogecoin\dogecoin.conf :

```dogecoin.conf
rpcuser=rpc_user
rpcpassword=rpc_password
rpcport=22555
txindex=1
rpccallip=127.0.0.1
server=1
```

#### Start Dogecoin Core:
```bash
dogecoind
```
Or open Dogecoin-Qt.
#### Verify:
```bash
dogecoin-cli -rpcuser=rpc_user -rpcpassword=rpc_password -rpcport=22555 getblockchaininfo
```

### Install IPFS Desktop:
Download from https://ipfs.io/#install.

#### Run IPFS Desktop or start the daemon:
```bash
& "C:\Program Files\IPFS Desktop\resources\app.asar.unpacked\node_modules\kubo\kubo\ipfs.exe" daemon
```

#### Verify:
```bash
ipfs id
```

### Configure doginals/.env:
#### Create/edit doginals/.env:
```.env
PROTOCOL_IDENTIFIER=D
NODE_RPC_URL=http://127.0.0.1:22555
NODE_RPC_USER=rpc_user
NODE_RPC_PASS=rpc_password
TESTNET=false
FEE_PER_KB=100000000
UNSPENT_API=https://unspent.dogeord.io/api/v2/address/unspent/
ORD=https://wonky-ord-v2.dogeord.io/
```

### (Optional) Configure `config.json`:
#### Create/edit `config.json` in the project root:
```json
{
  "external_apis": {
    "generic_llm": {
      "api_key": "your-api-key",
      "endpoint": "https://api.example.com/v1/completions"
    }
  },
  "storage": {
    "use_doginals": false
  }
}
```

## Usage

### Run the Server
Start the FastAPI server to handle AI queries, image/audio analysis, and storage:
```bash
cd C:\Users\<YourUsername>\OneDrive\Desktop\DogeSeekAI
python DogeSeekAIMain.py
```
*Expected output: Uvicorn running on http://0.0.0.0:8000*

### Run the Built-in Client
#### Run the Dogecoin-themed CLI:
```bash
python client/client.py
```

##### CLI Options:
1. Woof! Ask a question: Query about Dogecoin or anything else. *Example: What is Dogecoin?*

2. Bark! Upload audio: Transcribe audio files. *Example: Enter voice, then provide a file path (e.g., audio.wav).*

3. Howl! Upload image: Analyze images. *Example: Enter image, then provide a file path (e.g., image.jpg).*

4. Ruff! Store data: Store data on IPFS or Dogecoin (if enabled in config.json). *Example: Enter store, then provide data/file.*

5. Growl! Add API key: Configure an API key in config.json.Example: Enter api, then input your key and endpoint.

6. Enable! Doginals storage: Enable Doginals storage in config.json. *Example: Enter doginals, then confirm to enable.*

7. Prep! Set .env: Configure doginals/.env (must match dogecoin.conf). *Example: Enter env, then input rpc_user, rpc_pass, etc.*

8. Bone! Create/set wallet: Create a new wallet or set .wallet.json. *Example: Enter new or set, then follow prompts.

9. Fetch! Import wallet: Import wallet to Dogecoin Core.Example: Enter import, then provide private key.*

10. Sync! Sync wallet: Sync wallet with the blockchain.Example: Enter sync.

11. Split! Split UTXOs: Split UTXOs for bulk minting. *Example: Enter split, then specify count (e.g., 10).*

12. Send! Send funds: Send DOGE back after minting. *Example: Enter send, then specify address/amount.*

13. Moon! Mint to blockchain: Mint files, data, or delegates. *Example: Enter file (e.g., C:\doginals-main\DogeMeme.jpg, image/jpeg) or data (e.g., Woof!).*

14. Token! DRC-20 tokens: Deploy or mint DRC-20 tokens. *Example: Enter deploy (e.g., DOGECAC, 100000000) or mint.*

15. View! Check inscriptions: Start a server to view inscriptions.
*Example: Enter view, then provide inscription ID (e.g., 15f3b73df7e5c072becb1d84191843ba080734805addfccb650929719080f62e)*


### External APIText Query:
```bash
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"question": "What is AI?", "context": "AI is..."}'
```

#### Or in Python:
```python
import requests
response = requests.post("http://localhost:8000/api/predict", json={"question": "What is AI?", "context": "AI is..."})
print(response.json())
```

### Voice Input:
```bash
curl -X POST -F "file=@audio.wav" http://localhost:8000/api/voice
```

### Image Processing:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/api/image
```

### Storage:
```bash
curl -X POST -F "file=@data.json" -F "use_doginal=false" http://localhost:8000/api/upload_to_storage
```

### API Key Integration
***Add API keys via CLI (python client/client.py, option 5) or manually in config.json:***
```json
{
  "external_apis": {
    "generic_llm": {
      "api_key": "your-api-key",
      "endpoint": "https://api.example.com/v1/completions"
    }
  },
  "storage": {
    "use_doginals": false
  }
}
```

#### Storage GitHub: Code, configs, small datasets (<100MB), metadata (free).
IPFS: Large datasets, model weights (free, requires IPFS daemon).

#### Doginals: Immutable metadata/NFTs via doginals.js (costs DOGE, requires Node.js and Dogecoin Core).
Default: IPFS/GitHub. Enable Doginals in config.json via CLI option 6.


### Example Workflow
#### Start Services:
- Dogecoin Core: `dogecoind` or `Dogecoin-Qt`
- IPFS: `ipfs daemon`
- Server: `python DogeSeekAIMain.py`
- Run Client: `python client/client.py`
- Enable Doginals: Use `option 6` to set "use_doginals": true in `config.json`
- Set `.env`: Use `option 7` to configure `doginals/.env`.
- Create Wallet: Use `option 8` to create a new wallet and fund it.
- Import Wallet: Use `option 9` to import the wallet to Dogecoin Core.
- Sync Wallet: Use `option 10` to sync.
- Mint a File: Use `option 13` to mint `C:\doginals-main\DogeMeme.jpg`
- View Inscription: Use `option 15` to view at `http://localhost:3000/tx/<inscription_id>`.


## 🛡️ Security & Privacy 
DogeSeekAI runs 100% locally – your data stays on your computer!🔒

Encrypted Storage – The model is stored securely using End-to-End Encryption with AES-256🔑 

Prevents AI Attacks – Ensures safe updates & prevents hacking attempts🛡️

Dogecoin Node: Use rpcallowip=127.0.0.1 to restrict access to localhost🐶

IPFS: Configure firewall to allow private networks only to avoid malicious peers.

External APIs: Verify unspent.dogeord.io and wonky-ord-v2.dogeord.io in doginals/.env

Private Keys: Store doginals/.wallet.json securely and never share🔑

Updates: Keep Python, Node.js, Dogecoin Core, IPFS, and OS updated.


### GitHub Actions Automates:
- Dataset preprocessing and IPFS uploads.
- Model weight uploads to IPFS.
- Dependency updates with pipdeptree.
- Unit tests with pytest.
- README linting.
- Optional Doginal inscriptions (requires DOGE_ADDRESS, DOGE_RPC secrets in GitHub).
- Defined in .github/workflows/preprocess.yml, triggers daily or on push to main.

## 🤝 Contributions
Contribute via PRs or donate Dogecoin to `GreatApe42069`:

***Doge Address: D9pqzxiiUke5eodEzMmxZAxpFcbvwuM4Hg***

***GitHub Profile: @GreatApe42069***

***X.com Profile: @GreatApe42069E***

## License
`DogeSeekAI` is licensed under the Creative Commons Zero v1.0 Universal. The Creative Commons CC0 Public Domain Dedication waives copyright interest in a work you've created and dedicates it to the worldwide public domain. Use CC0 to opt out of copyright entirely and ensure your work has the widest reach. As with the Unlicense and typical software licenses, CC0 disclaims warranties. CC0 is very similar to the Unlicense. 

## Extra Implementation Details:
### Interactive Prompt: 
- Option 5 in client/client.py prompts for API key and endpoint, updating `config.json`.
- Option 6 enables Doginals storage in `config.json`.
- Options 7–15 handle Dogecoin wallet setup, minting, and viewing (see doginals/doginalsReadme.md).
- Users can skip inputs (press Enter) to avoid changes.
- Doginals Integration: `Uses doginals.js` for blockchain inscriptions (files, data, DRC-20 tokens).
- Error-Free: Consistent imports, Python 3.8+ compatibility, valid JSON/YAML syntax.
-Dogecoin Theme: CLI aligns with Dogecoin Community branding.

## Troubleshooting
### Server Errors:
- Check `DogeSeekAIMain.py` output
- Ensure Python dependencies: `pip install -r requirements.txt`
### Client Errors:
- Debug JSON errors with Raw response in `client.py`
- Verify server is running: curl http://localhost:8000

### Doginals Errors:
- Test: node doginals.js wallet new in doginals folder.
- Ensure `.env` matches `dogecoin.conf` and wallet is funded.

### IPFS Errors:
- Verify daemon: ipfs id

## Next Steps To Do
- Test CLI: Run python `client/client.py`, test options 1–15, especially option 6 for Doginals storage.
- Verify Config: Check `config.json` after enabling Doginals.
- Check Implementation of `doginals.js` Wrapper: Use subprocess in `doginals/inscribe.py` (already implemented).
- Community: 🐶Rally the Dogecoin Community on GitHub and X!🚀

***DogeSeekAI empowers users to supercharge their AI experience, keeping it Decentralized, End-to-End Encrypted, Open-source, Multimodal, and Doge-tastic. Much wow, let’s keep building!***
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/a069061d-af2e-405f-8e4b-6fe999c3bfb4" />


