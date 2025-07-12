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

    def update_api_key(self, api_key, endpoint):
        """Update config.json with user-provided API key and endpoint."""
        self.config["external_apis"]["generic_llm"]["api_key"] = api_key
        self.config["external_apis"]["generic_llm"]["endpoint"] = endpoint
        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=4)
        print("API key and endpoint updated! DogeSeekAI is supercharged! 🚀")

    def run(self):
        print("""
        ██████╗  ██████╗  ██████╗ ███████╗███████╗███████╗███████╗██╗  ██╗ █████╗ ██╗
        ██╔══██╗██╔═══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝██╔════╝██║ ██╔╝██╔══██╗██║
        ██║  ██║██║   ██║██║  ███╗█████╗  █████╗  █████╗  █████╗  █████╔╝ ███████║██║
        ██║  ██║██║   ██║██║   ██║██╔══╝      ██║ ██╔══╝  ██╔══╝  ██╔═██╗ ██╔══██║██║
        ██████╔╝╚██████╔╝╚██████╔╝███████╗██████║ ███████╗███████╗██║  ██╗██║  ██║██║
        ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                  🐶 DogeSeekAI CLI - Much Wow, Such AI! 🐶
        """)
        print("🐶 Welcome to DogeSeekAI! Much wow, such AI! 🐶")
        if not self.config["external_apis"]["generic_llm"]["api_key"]:
            print("Tip: Add an API key in config.json or use option 5 to supercharge DogeSeekAI!")
        
        while True:
            print("\nPick an action:")
            for i, prompt in enumerate(self.prompts, 1):
                print(f"{i}. {prompt}")
            print("0. Exit")
            
            choice = input("Enter number (0-5): ")
            if choice == "0":
                print("So long, Doge friend! 🐾")
                break
            elif choice == "1":
                question = input("Enter your question: ")
                context = input("Optional context (press Enter to skip): ")
                response = requests.post(f"{self.base_url}/api/predict", json={"question": question, "context": context})
                print(f"DogeSeekAI says: {response.json()['answer']}")
            elif choice == "2":
                file_path = input("Enter audio file path (e.g., audio.wav): ")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        response = requests.post(f"{self.base_url}/api/voice", files={"file": f})
                    print(f"Transcription: {response.json()['transcription']}\nEmotion: {response.json()['emotion']}")
                else:
                    print("File not found! Try again, pup!")
            elif choice == "3":
                file_path = input("Enter image file path (e.g., image.jpg): ")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        response = requests.post(f"{self.base_url}/api/image", files={"file": f})
                    print(f"Image features extracted: {len(response.json()['features'])} dimensions")
                else:
                    print("File not found! Try again, pup!")
            elif choice == "4":
                file_path = input("Enter file path to store (e.g., data.json): ")
                use_doginal = input("Inscribe on Dogecoin? (y/n): ").lower() == "y"
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        response = requests.post(f"{self.base_url}/api/upload_to_storage", files={"file": f}, data={"use_doginal": str(use_doginal).lower()})
                    print(f"Storage info: {response.json()}")
                else:
                    print("File not found! Try again, pup!")
            elif choice == "5":
                api_key = input("Enter your API key (press Enter to skip): ")
                endpoint = input("Enter API endpoint (e.g., https://api.example.com/v1/completions, press Enter to skip): ")
                if api_key and endpoint:
                    self.update_api_key(api_key, endpoint)
                else:
                    print("No API key or endpoint provided. You can still edit config.json manually!")
            else:
                print("Invalid choice, try again! Woof!")

if __name__ == "__main__":
    client = DogeSeekAIClient()
    client.run()
