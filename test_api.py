"""Quick test for the NIDS API."""
import requests
import json

base = "http://127.0.0.1:8888"

# 1. Health check
print("=== Health Check ===")
r = requests.get(f"{base}/health")
print(json.dumps(r.json(), indent=2))

# 2. Get features
print("\n=== Features ===")
r = requests.get(f"{base}/features")
data = r.json()
print(f"Count: {data['count']}")
print(f"First 5: {data['features'][:5]}")

# 3. Predict (zeros = likely benign)
print("\n=== Prediction (zeros → likely benign) ===")
r = requests.post(f"{base}/predict", json={"features": [0.0] * 31})
print(json.dumps(r.json(), indent=2))

# 4. Predict (high values = likely attack)
print("\n=== Prediction (high values → likely attack) ===")
r = requests.post(f"{base}/predict", json={"features": [100.0] * 31})
print(json.dumps(r.json(), indent=2))

print("\n✓ All endpoints working!")
