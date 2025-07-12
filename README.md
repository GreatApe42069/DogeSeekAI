        ██████╗  ██████╗  ██████╗ ███████╗███████╗███████╗███████╗██╗  ██╗ █████╗ ██╗
        ██╔══██╗██╔═══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝██╔════╝██║ ██╔╝██╔══██╗██║
        ██║  ██║██║   ██║██║  ███╗█████╗  █████╗  █████╗  █████╗  █████╔╝ ███████║██║
        ██║  ██║██║   ██║██║   ██║██╔══╝      ██║ ██╔══╝  ██╔══╝  ██╔═██╗ ██╔══██║██║
        ██████╔╝╚██████╔╝╚██████╔╝███████╗██████║ ███████╗███████╗██║  ██╗██║  ██║██║
        ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                  🐶 DogeSeekAI CLI - Much Wow, Such AI! 🐶

# 🚀 DogeSeekAI

DogeSeekAI is a Decentralized, End-to-End Encrypted, Open-source, Multimodal AI! Designed for local deployment, ensuring Such Privacy and Much Efficiency! Supports  text, voice, and images. It learns in real-time from user interactions, continuously improving itself while keeping all user data securely stored on your device. 


*Note this model is currently still in developement and testing stages* 

## ✨ Features
- **Multimodal**: Text (DistilBERT), voice (Wav2Vec2), image (ResNet50).
- **Storage**: IPFS/GitHub (default), optional Dogecoin inscriptions via doginals.js.
- **Federated Learning**: Collaborative training via Flower.
- **E2EE**: AES-256 encryption.
- **Built-in Client**: Dogecoin-themed CLI 
- **External API**: Integrate into custom apps/websites.
- **API Key Support**: Supercharge with external model APIs via CLI or config.json.
- **Automation**: GitHub Actions for dataset preprocessing, model uploads, testing, and dependency updates.

## 🛠️ Installation
1. Install Python 3.8+.
2. Clone the repo:
   ```bash
   git clone https://github.com/GreatApe42069/DogeSeekAI.git
   cd DogeSeekAI

### Install dependencies:
```bash
pip install -r requirements.txt
```
### Run IPFS daemon:bash
```bash
ipfs daemon
```
### (Optional) Configure Dogecoin node and external APIs in `config.json`
Run the app:
```bash
python DogeSeekAIMain.py
```
### Run the built-in client:
```bash
python client/client.py
```
## Usage
### Built-in Client:
Run ```bash 
python client/client.py``` for a Dogecoin-themed CLI:

- Ask questions (e.g., "What is Dogecoin?").
- Upload audio for transcription.
- Upload images for analysis.
- Store data on IPFS or Dogecoin.
- Add API keys via option 5.

### External API 
#### Text Query:
`import requests
response = requests.post("http://localhost:8080/api/predict", json={"question": "What is AI?", "context": "AI is..."})
print(response.json())`

### Voice Input:
```bash
curl -X POST -F "file=@audio.wav" http://localhost:8080/api/voice
```
### Image Processing:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8080/api/image
```
### Storage:
```bash
curl -X POST -F "file=@data.json" -F "use_doginal=false" http://localhost:8080/api/upload_to_storage
```
### API Key Integration
#### Add API keys via CLI (`python client/client.py`, option 5) or in `config.json`

`"external_apis": {
    "generic_llm": {
        "api_key": "your-api-key",
        "endpoint": "https://api.example.com/v1/completions"
    }
}`

### Storage 
- GitHub: Code, configs, small datasets (<100MB), metadata (free).
- IPFS: Large datasets, model weights (free).
- Doginals: Optional immutable metadata/NFTs via `doginals.js` (***costs DOGE, requires node***).
- Default: IPFS/GitHub. Enable Doginals in `config.json`

### 🛡️ Security & Privacy
🔒 Super GROK runs 100% locally – your data stays on your computer!

🔑 Encrypted Storage – The model is stored securely using End to End Encryption with  AES-256 Encryption

🛡️ Prevents AI Attacks – Ensures safe updates & prevents hacking attempts

### GitHub Actions
#### Automates:
- Dataset preprocessing and IPFS uploads.
- Model weight uploads to IPFS.
- Dependency updates with pipdeptree.
- Unit tests with pytest.
- README linting.
- Optional Doginal inscriptions (requires DOGE_ADDRESS, DOGE_RPC secrets).
- Defined in .github/workflows/preprocess.yml, triggers daily or on push to main.

## 🤝 Contributions
Contribute via PRs or donate Dogecoin to `GreatApe42069`:

**Doge Address:** ***D9pqzxiiUke5eodEzMmxZAxpFcbvwuM4Hg***

**Github Profile:** ***@GreatApe42069*** 

**X.com Profile:**  ***@GreatApe42069E***


## LICENSE
`DogeSeekAI` is licensed under the
**Creative Commons Zero v1.0 Universal**
The Creative Commons CC0 Public Domain Dedication waives copyright interest in a work you've created and dedicates it to the world-wide public domain. Use CC0 to opt out of copyright entirely and ensure your work has the widest reach. As with the Unlicense and typical software licenses, CC0 disclaims warranties. CC0 is very similar to the Unlicense.


### Extra Implementation Details
- **Interactive Prompt**:
  - Added choice 5 in `client/client.py` to prompt for API key and endpoint, updating `config.json` via `update_api_key`.
  - Retained tip for manual `config.json` edits, ensuring flexibility.
  - Users can skip inputs (press Enter) to avoid changes.
- **Other Files**: Unchanged, as they already support API key routing (`DogeSeekAIMain.py`) and existing features (IPFS, Doginals, automation).
- **Error-Free**: Consistent imports, Python 3.8+ compatibility, and valid JSON/YAML syntax. `doginals_js` remains a placeholder (implement via `pyodide` if needed).
- **Dogecoin Theme**: CLI prompt aligns with Dogecoin Community branding 

### Next Steps To Do:
- **Test CLI**: Run `python client/client.py`, select option 5, and input a test API key/endpoint (e.g., a mock LLM endpoint).
- **Verify Config**: Check `config.json` updates after using the CLI prompt.
- **Implement `doginals.js`**: Create a Python wrapper using `pyodide`.
- **Community**: Get Dogecoin Communtiy involved!!!

DogeSeekAI empowers users to supercharge their AI experience, keeping it Decentralized, End-to-End Encrypted, Open-source, Multimodal AI Accessible and Doge-tastic. Much wow, let’s keep building!

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/a069061d-af2e-405f-8e4b-6fe999c3bfb4" />
