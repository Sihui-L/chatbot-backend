from fastapi.testclient import TestClient

from src.main import application

client = TestClient(application)

def test_websocket_endpoint():
    with client.websocket_connect("/api/v1/ws") as websocket:
        pass