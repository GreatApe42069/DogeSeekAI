import subprocess

class DoginalInscriber:
    def __init__(self, wallet_address, rpc_url, private_key=None):
        self.address = address
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.node_path = "node"
        self.script_path = "doginals/doginals.js"  ss# Adjust to "doginals/doginals.js" if in subfolder

    def inscribe(self, data, content_type="text/plain;charset=utf-8"):
        if isinstance(data, bytes):
            hex_data = data.hex()
        else:
            hex_data = data.encode('utf-8').hex()

        cmd = [
            self.node_path,
            self.script_path,
            "mint",
            self.wallet_address,
            content_type,
            hex_data
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd="C:/Users/Ol Soles/OneDrive/Desktop/DogeSeekAI/doginals",
                capture_output=True,
                text=True,
                check=True
            )
            print("Inscription Output:", result.stdout)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise Exception(f"Inscription failed: {e.stderr}")

    def inscribe_file(self, file_path, content_type):
        with open(file_path, 'rb') as f:
            file_data = f.read()
        return self.inscribe(file_data, content_type)

    def inscribe_delegate(self, inscription_id):
        cmd = [
            self.node_path,
            self.script_path,
            "mint",
            self.wallet_address,
            "",
            "",
            inscription_id
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd="C:/Users/Ol Soles/OneDrive/Desktop/DogeSeekAI/doginals",
                capture_output=True,
                text=True,
                check=True
            )
            print("Delegate Inscription Output:", result.stdout)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise Exception(f"Delegate inscription failed: {e.stderr}")

    def deploy_drc20(self, ticker, total, max_mint):
        cmd = [
            self.node_path,
            self.script_path,
            "drc-20",
            "deploy",
            self.wallet_address,
            ticker,
            str(total),
            str(max_mint)
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd="C:/Users/Ol Soles/OneDrive/Desktop/DogeSeekAI/doginals",
                capture_output=True,
                text=True,
                check=True
            )
            print("DRC-20 Deploy Output:", result.stdout)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise Exception(f"DRC-20 deploy failed: {e.stderr}")

    def mint_drc20(self, ticker, amount):
        cmd = [
            self.node_path,
            self.script_path,
            "drc-20",
            "mint",
            self.wallet_address,
            ticker,
            str(amount)
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd="C:/Users/Ol Soles/OneDrive/Desktop/DogeSeekAI/doginals",
                capture_output=True,
                text=True,
                check=True
            )
            print("DRC-20 Mint Output:", result.stdout)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise Exception(f"DRC-20 mint failed: {e.stderr}")

if __name__ == "__main__":
    inscriber = DoginalInscriber("D9UcJkdirVLY11UtF77WnC8peg6xRYsogu", "http://127.0.0.1:22555")
    tx_output = inscriber.inscribe("Woof!")
    print(f"Inscription result: {tx_output}")
