# test_api.py
import requests
import json

# Test the API
base_url = "http://localhost:8000"  # or your AWS URL

# Health check
response = requests.get(f"{base_url}/health")
print("Health:", response.json())

# Chat request
chat_data = {
    "message": "Write a Python function to calculate fibonacci numbers",
    "max_tokens": 500,
    "temperature": 0.7
}

response = requests.post(f"{base_url}/chat", json=chat_data)
print("Chat response:", response.json())