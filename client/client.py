import requests
import json
import os

class DogeSeekAIClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.prompts = [
            "Woof! Ask me a question about Dogecoin or anything else! (e.g., 'What is Dogecoin?')",
            "Bark! Upload an audio file for transcription! (Type 'voice' to start)",
            "Howl! Upload an image for analysis! (Type 'image' to start)",
            "Ruff! Want to store data on IPFS or Dogecoin? (Type 'store' to start)",
            "Growl! Add an API key to supercharge DogeSeekAI! (Type 'api' to start)"
        ]
        with open("config.json", "r") as f:
            self.config = json.load(f)
        # Dogecoin yellow (hex #FFC107, RGB 255, 193, 7)
        self.yellow = "\033[38;2;255;193;7m"
        self.reset = "\033[0m"

    def update_api_key(self, api_key, endpoint):
        """Update config.json with user-provided API key and endpoint."""
        self.config["external_apis"]["generic_llm"]["api_key"] = api_key
        self.config["external_apis"]["generic_llm"]["endpoint"] = endpoint
        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=4)
        print(f"{self.yellow}API key and endpoint updated! DogeSeekAI is supercharged! 🚀{self.reset}")

    def run(self):
        print(f"{self.yellow}")
        print("""
        ██████╗  ██████╗  ██████╗ ███████╗███████╗███████╗███████╗██╗  ██╗ █████╗ ██╗
        ██╔══██╗██╔═══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝██╔════╝██║ ██╔╝██╔══██╗██║
        ██║  ██║██║   ██║██║  ███╗█████╗  █████╗  █████╗  █████╗  █████╔╝ ███████║██║
        ██║  ██║██║   ██║██║   ██║██╔══╝      ██║ ██╔══╝  ██╔══╝  ██╔═██╗ ██╔══██║██║
        ██████╔╝╚██████╔╝╚██████╔╝███████╗██████║ ███████╗███████╗██║  ██╗██║  ██║██║
        ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                  🐶 DogeSeekAI CLI - Much Wow, Such AI! 🐶
        """)
        print(f"🐶 Welcome to DogeSeekAI! Much wow, such AI! 🐶")
        if not self.config["external_apis"]["generic_llm"]["api_key"]:
            print("Tip: Add an API key in config.json or use option 5 to supercharge DogeSeekAI!")
        
        while True:
            print("\nPick an action:")
            for i, prompt in enumerate(self.prompts, 1):
                print(f"{i}. {prompt}")
            print("0. Exit")
            
            choice = input(f"{self.yellow}Enter number (0-5): {self.reset}")
            if choice == "0":
                print(f"{self.yellow}So long, Doge friend! 🐾{self.reset}")
                break
            elif choice == "1":
                question = input(f"{self.yellow}Enter your question: {self.reset}")
                context = input(f"{self.yellow}Optional context (press Enter to skip): {self.reset}")
                response = requests.post(f"{self.base_url}/api/predict", json={"question": question, "context": context})
                print(f"{self.yellow}DogeSeekAI says: {response.json()['answer']}{self.reset}")
            elif choice == "2":
                file_path = input(f"{self.yellow}Enter audio file path (e.g., audio.wav): {self.reset}")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        response = requests.post(f"{self.base_url}/api/voice", files={"file": f})
                    print(f"{self.yellow}Transcription: {response.json()['transcription']}\nEmotion: {response.json()['emotion']}{self.reset}")
                else:
                    print(f"{self.yellow}File not found! Try again, pup!{self.reset}")
            elif choice == "3":
                file_path = input(f"{self.yellow}Enter image file path (e.g., image.jpg): {self.reset}")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        response = requests.post(f"{self.base_url}/api/image", files={"file": f})
                    print(f"{self.yellow}Image features extracted: {len(response.json()['features'])} dimensions{self.reset}")
                else:
                    print(f"{self.yellow}File not found! Try again, pup!{self.reset}")
            elif choice == "4":
                file_path = input(f"{self.yellow}Enter file path to store (e.g., data.json): {self.reset}")
                use_doginal = input(f"{self.yellow}Inscribe on Dogecoin? (y/n): {self.reset}").lower() == "y"
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        response = requests.post(f"{self.base_url}/api/upload_to_storage", files={"file": f}, data={"use_doginal": str(use_doginal).lower()})
                    print(f"{self.yellow}Storage info: {response.json()}{self.reset}")
                else:
                    print(f"{self.yellow}File not found! Try again, pup!{self.reset}")
            elif choice == "5":
                api_key = input(f"{self.yellow}Enter your API key (press Enter to skip): {self.reset}")
                endpoint = input(f"{self.yellow}Enter API endpoint (e.g., https://api.example.com/v1/completions, press Enter to skip): {self.reset}")
                if api_key and endpoint:
                    self.update_api_key(api_key, endpoint)
                else:
                    print(f"{self.yellow}No API key or endpoint provided. You can still edit config.json manually!{self.reset}")
            else:
                print(f"{self.yellow}Invalid choice, try again! Woof!{self.reset}")
        print(f"{self.reset}")

if __name__ == "__main__":
    client = DogeSeekAIClient()
    client.run()
