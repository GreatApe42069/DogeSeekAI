import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("sys.path:", sys.path)  # Debug line
import requests
import json
from doginals.inscribe import DoginalInscriber
import subprocess

class DogeSeekAIClient:
    def __init__(self):
        self.yellow = "\033[38;2;255;193;7m"
        self.reset = "\033[0m"
        self.doginals_dir = "C:/Users/Ol Soles/OneDrive/Desktop/DogeSeekAI/doginals"
        self.config_path = "C:/Users/Ol Soles/OneDrive/Desktop/DogeSeekAI/config.json"
        # Load config
        with open(self.config_path, "r") as config_file:
            config = json.load(config_file)
        self.base_url = f"http://localhost:{config['port']}"
        # Load wallet data
        wallet_path = os.path.join(self.doginals_dir, ".wallet.json")
        doge_address = ""
        private_key = ""
        if os.path.exists(wallet_path):
            with open(wallet_path, "r") as wallet_file:
                wallet_data = json.load(wallet_file)
                doge_address = wallet_data.get("address", "")
                private_key = wallet_data.get("private_key", "")
        self.inscriber = DoginalInscriber(doge_address, config["doge_rpc"], private_key=private_key)s

    def run(self):
        print(f"""
{self.yellow}
        ██████╗  ██████╗  ██████╗ ███████╗███████╗███████╗███████╗██╗  ██╗ █████╗ ██╗
        ██╔══██╗██╔═══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝██╔════╝██║ ██╔╝██╔══██╗██║
        ██║  ██║██║   ██║██║  ███╗█████╗  █████╗  █████╗  █████╗  █████╔╝ ███████║██║
        ██║  ██║██║   ██║██║   ██║██╔══╝      ██║ ██╔══╝  ██╔══╝  ██╔═██╗ ██╔══██║██║
        ██████╔╝╚██████╔╝╚██████╔╝███████╗██████║ ███████╗███████╗██║  ██╗██║  ██║██║
        ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                  🐶 DogeSeekAI CLI - Much Wow, Such AI! 🐶
{self.reset}
🐶 Welcome to DogeSeekAI! Much wow, such AI! 🐶
Tip: Add an API key in config.json or use option 5 to supercharge DogeSeekAI!
""")
        while True:
            print("""
Pick an action:
1. Woof! Ask me a question about Dogecoin or anything else! (e.g., 'What is Dogecoin?')
2. Bark! Upload an audio file for transcription! (Type 'voice' to start)
3. Howl! Upload an image for analysis! (Type 'image' to start)
4. Ruff! Want to store data on IPFS or Dogecoin? (Type 'store' to start)
5. Growl! Add an API key to supercharge DogeSeekAI! (Type 'api' to start)
6. Enable! Enable Doginals storage in config.json! (Type 'doginals' to start)
7. Prep! Set up your .env file for Dogecoin node! (Type 'env' to start)
8. Bone! Create or set your Dogecoin wallet! (Type 'wallet' to start)
9. Fetch! Import wallet to Dogecoin node! (Type 'import' to start)
10. Sync! Sync your wallet with the blockchain! (Type 'sync' to start)
11. Split! Split UTXOs for bulk minting! (Type 'split' to start)
12. Send! Send funds back after minting! (Type 'send' to start)
13. Moon! Mint files or data to the Dogecoin blockchain! (Type 'mint' to start)
14. Token! Create or mint DRC-20 tokens! (Type 'token' to start)
15. View! Check your Doge creations on the blockchain! (Type 'view' to start)
0. Exit
""")
            choice = input(f"{self.yellow}Enter number (0-15): {self.reset}")
            if choice == "0":
                print(f"{self.yellow}Such bye, much wow!{self.reset}")
                break
            elif choice == "1":
                question = input(f"{self.yellow}Enter your question: {self.reset}")
                context = input(f"{self.yellow}Optional context (press Enter to skip): {self.reset}")
                try:
                    response = requests.post(f"{self.base_url}/ask", json={"question": question, "context": context})
                    print("Raw response:", response.text)
                    print(f"{self.yellow}DogeSeekAI says: {response.json()['answer']}{self.reset}")
                except Exception as e:
                    print(f"{self.yellow}Error: {str(e)}{self.reset}")
            elif choice == "2":
                print(f"{self.yellow}Bark! Enter 'voice' to upload audio!{self.reset}")
                # Existing audio logic
            elif choice == "3":
                print(f"{self.yellow}Howl! Enter 'image' to upload an image!{self.reset}")
                # Existing image logic
            elif choice == "4":
                print(f"{self.yellow}Ruff! Enter 'store' to store on IPFS!{self.reset}")
                # Existing IPFS logic
            elif choice == "5":
                print(f"{self.yellow}Growl! Enter 'api' to add an API key!{self.reset}")
                # Existing API key logic
            elif choice == "6":
                print(f"{self.yellow}Enable! Enable Doginals storage in config.json!{self.reset}")
                doginals_choice = input(f"{self.yellow}Enter 'doginals' to enable Doginals storage (costs DOGE): {self.reset}")
                if doginals_choice == "doginals":
                    try:
                        config = {"external_apis": {"generic_llm": {"api_key": "", "endpoint": ""}}, "storage": {"use_doginals": False}}
                        if os.path.exists(self.config_path):
                            with open(self.config_path, "r") as f:
                                config = json.load(f)
                        config["storage"]["use_doginals"] = True
                        with open(self.config_path, "w") as f:
                            json.dump(config, f, indent=4)
                        print(f"{self.yellow}Doginals storage enabled in config.json! Such blockchain, much wow!{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}Failed to update config.json: {str(e)}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'doginals'!{self.reset}")
            elif choice == "7":
                print(f"{self.yellow}Prep! Set up your .env file for Dogecoin node!{self.reset}")
                env_choice = input(f"{self.yellow}Enter 'env' to set .env (must match dogecoin.conf): {self.reset}")
                if env_choice == "env":
                    rpc_user = input(f"{self.yellow}Enter RPC username (must match dogecoin.conf): {self.reset}")
                    rpc_pass = input(f"{self.yellow}Enter RPC password (must match dogecoin.conf): {self.reset}")
                    rpc_url = input(f"{self.yellow}Enter RPC URL (default: http://127.0.0.1:22555): {self.reset}") or "http://127.0.0.1:22555"
                    fee = input(f"{self.yellow}Enter fee per KB (default: 60000000): {self.reset}") or "60000000"
                    env_content = f"""
PROTOCOL_IDENTIFIER=D
NODE_RPC_URL={rpc_url}
NODE_RPC_USER={rpc_user}
NODE_RPC_PASS={rpc_pass}
TESTNET=false
FEE_PER_KB={fee}
UNSPENT_API=https://unspent.dogeord.io/api/v2/address/unspent/
ORD=https://wonky-ord-v2.dogeord.io/
"""
                    try:
                        with open(os.path.join(self.doginals_dir, ".env"), "w") as f:
                            f.write(env_content.strip())
                        print(f"{self.yellow}.env file updated! Such config, much wow!{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}Failed to write .env: {str(e)}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'env'!{self.reset}")
            elif choice == "8":
                print(f"{self.yellow}Bone! Create or set your Dogecoin wallet!{self.reset}")
                wallet_choice = input(f"{self.yellow}Enter 'new' to create a wallet or 'set' to input one: {self.reset}")
                if wallet_choice == "new":
                    try:
                        result = subprocess.run(
                            ["node", "doginals.js", "wallet", "new"],
                            cwd=self.doginals_dir,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        print(f"{self.yellow}New wallet created! {result.stdout}{self.reset}")
                        print(f"{self.yellow}Check .wallet.json in doginals folder! Fund the address shown!{self.reset}")
                    except subprocess.CalledProcessError as e:
                        print(f"{self.yellow}Wallet creation failed: {e.stderr}{self.reset}")
                elif wallet_choice == "set":
                    address = input(f"{self.yellow}Enter Dogecoin address: {self.reset}")
                    priv_key = input(f"{self.yellow}Enter private key: {self.reset}")
                    try:
                        with open(os.path.join(self.doginals_dir, ".wallet.json"), "w") as f:
                            json.dump({"address": address, "private_key": priv_key}, f)
                        print(f"{self.yellow}.wallet.json updated! Such wallet, much wow!{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}Failed to write .wallet.json: {str(e)}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'new' or 'set'!{self.reset}")
            elif choice == "9":
                print(f"{self.yellow}Fetch! Import wallet to Dogecoin node!{self.reset}")
                import_choice = input(f"{self.yellow}Enter 'import' to import private key: {self.reset}")
                if import_choice == "import":
                    priv_key = input(f"{self.yellow}Enter private key from .wallet.json: {self.reset}")
                    label = input(f"{self.yellow}Enter wallet label (e.g., DogeSeekAIWallet): {self.reset}")
                    try:
                        result = subprocess.run(
                            ["dogecoin-cli", "importprivkey", priv_key, label, "false"],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        print(f"{self.yellow}Wallet imported! {result.stdout}{self.reset}")
                        print(f"{self.yellow}Fund the wallet address in Dogecoin-Qt or CLI!{self.reset}")
                    except subprocess.CalledProcessError as e:
                        print(f"{self.yellow}Import failed: {e.stderr}{self.reset}")
                        print(f"{self.yellow}Ensure Dogecoin node is running and dogecoin-cli is in PATH!{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'import'!{self.reset}")
            elif choice == "10":
                print(f"{self.yellow}Sync! Sync your wallet with the blockchain!{self.reset}")
                sync_choice = input(f"{self.yellow}Enter 'sync' to sync wallet: {self.reset}")
                if sync_choice == "sync":
                    try:
                        result = subprocess.run(
                            ["node", "doginals.js", "wallet", "sync"],
                            cwd=self.doginals_dir,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        print(f"{self.yellow}Wallet synced! {result.stdout}{self.reset}")
                    except subprocess.CalledProcessError as e:
                        print(f"{self.yellow}Sync failed: {e.stderr}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'sync'!{self.reset}")
            elif choice == "11":
                print(f"{self.yellow}Split! Split UTXOs for bulk minting!{self.reset}")
                split_choice = input(f"{self.yellow}Enter 'split' to split UTXOs: {self.reset}")
                if split_choice == "split":
                    count = input(f"{self.yellow}Enter number of UTXOs to split (e.g., 10): {self.reset}")
                    try:
                        result = subprocess.run(
                            ["node", "doginals.js", "wallet", "split", count],
                            cwd=self.doginals_dir,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        print(f"{self.yellow}UTXOs split! {result.stdout}{self.reset}")
                    except subprocess.CalledProcessError as e:
                        print(f"{self.yellow}Split failed: {e.stderr}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'split'!{self.reset}")
            elif choice == "12":
                print(f"{self.yellow}Send! Send funds back after minting!{self.reset}")
                send_choice = input(f"{self.yellow}Enter 'send' to send funds: {self.reset}")
                if send_choice == "send":
                    address = input(f"{self.yellow}Enter destination address: {self.reset}")
                    amount = input(f"{self.yellow}Enter amount (optional, press Enter to skip): {self.reset}")
                    cmd = ["node", "doginals.js", "wallet", "send", address]
                    if amount:
                        cmd.append(amount)
                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=self.doginals_dir,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        print(f"{self.yellow}Funds sent! {result.stdout}{self.reset}")
                    except subprocess.CalledProcessError as e:
                        print(f"{self.yellow}Send failed: {e.stderr}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'send'!{self.reset}")
            elif choice == "13":
                print(f"{self.yellow}Moon! Mint files or data to the Dogecoin blockchain!{self.reset}")
                mint_type = input(f"{self.yellow}Enter 'file' to mint a file, 'data' for raw data, or 'delegate' for a delegate: {self.reset}")
                if mint_type == "file":
                    file_path = input(f"{self.yellow}Enter file path (e.g., C:/doginals-main/DogeMeme.jpg): {self.reset}")
                    content_type = input(f"{self.yellow}Enter content type (e.g., image/jpeg, audio/mpeg, text/plain): {self.reset}")
                    try:
                        tx_output = self.inscriber.inscribe_file(file_path, content_type)
                        print(f"{self.yellow}Inscription result: {tx_output}{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}Minting failed: {str(e)}{self.reset}")
                elif mint_type == "data":
                    data = input(f"{self.yellow}Enter data to mint (e.g., Woof!): {self.reset}")
                    content_type = input(f"{self.yellow}Enter content type (default: text/plain;charset=utf-8): {self.reset}") or "text/plain;charset=utf-8"
                    try:
                        tx_output = self.inscriber.inscribe(data, content_type)
                        print(f"{self.yellow}Inscription result: {tx_output}{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}Minting failed: {str(e)}{self.reset}")
                elif mint_type == "delegate":
                    inscription_id = input(f"{self.yellow}Enter delegate inscription ID (e.g., 15f3b73df7e5c072becb1d84191843ba080734805addfccb650929719080f62e): {self.reset}")
                    try:
                        tx_output = self.inscriber.inscribe_delegate(inscription_id)
                        print(f"{self.yellow}Delegate inscription result: {tx_output}{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}Delegate minting failed: {str(e)}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'file', 'data', or 'delegate'!{self.reset}")
            elif choice == "14":
                print(f"{self.yellow}Token! Create or mint DRC-20 tokens!{self.reset}")
                token_type = input(f"{self.yellow}Enter 'deploy' to create a token or 'mint' to mint tokens: {self.reset}")
                if token_type == "deploy":
                    ticker = input(f"{self.yellow}Enter token ticker (e.g., DOGECAC): {self.reset}")
                    total = input(f"{self.yellow}Enter total supply (e.g., 100000000): {self.reset}")
                    max_mint = input(f"{self.yellow}Enter max mint per tx (e.g., 100000000): {self.reset}")
                    try:
                        tx_output = self.inscriber.deploy_drc20(ticker, total, max_mint)
                        print(f"{self.yellow}DRC-20 deploy result: {tx_output}{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}DRC-20 deploy failed: {str(e)}{self.reset}")
                elif token_type == "mint":
                    ticker = input(f"{self.yellow}Enter token ticker (e.g., DOGE): {self.reset}")
                    amount = input(f"{self.yellow}Enter amount to mint (e.g., 10000000): {self.reset}")
                    try:
                        tx_output = self.inscriber.mint_drc20(ticker, amount)
                        print(f"{self.yellow}DRC-20 mint result: {tx_output}{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}DRC-20 mint failed: {str(e)}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'deploy' or 'mint'!{self.reset}")
            elif choice == "15":
                print(f"{self.yellow}View! Check your Doge creations on the blockchain!{self.reset}")
                view_choice = input(f"{self.yellow}Enter 'view' to start the server: {self.reset}")
                if view_choice == "view":
                    inscription_id = input(f"{self.yellow}Enter inscription ID (e.g., 15f3b73df7e5c072becb1d84191843ba080734805addfccb650929719080f62e): {self.reset}")
                    try:
                        print(f"{self.yellow}Starting Doginals server... Open http://localhost:3000/tx/{inscription_id} in your browser!{self.reset}")
                        subprocess.Popen(
                            ["node", "doginals.js", "server"],
                            cwd=self.doginals_dir,
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                        )
                        print(f"{self.yellow}Server started! Check your Doge masterpiece!{self.reset}")
                    except Exception as e:
                        print(f"{self.yellow}Server start failed: {str(e)}{self.reset}")
                else:
                    print(f"{self.yellow}Invalid choice, try 'view'!{self.reset}")
            else:
                print(f"{self.yellow}Invalid choice, try again! Woof!{self.reset}")

if __name__ == "__main__":
    client = DogeSeekAIClient()
    client.run()