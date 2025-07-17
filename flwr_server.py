import flwr as flwr

if __name__ == "__main__":
    flwr.server.start_server(
        server_address="localhost:8083",
        config=flwr.server.ServerConfig(num_rounds=3)
    )