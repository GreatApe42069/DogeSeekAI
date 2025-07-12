import json
from doginals_js import DoginalClient

class DoginalInscriber:
    def __init__(self, wallet_address, rpc_url):
        self.client = DoginalClient(wallet_address=wallet_address, rpc_url=rpc_url)
    
    def inscribe(self, data, parent_id=""):
        try:
            txid = self.client.inscribe(data, parent_id=parent_id)
            return txid
        except Exception as e:
            raise Exception(f"Inscription failed: {str(e)}")
    
    def fetch_inscription(self, txid):
        try:
            return self.client.get_inscription_data(txid)
        except Exception as e:
            raise Exception(f"Fetch failed: {str(e)}")
