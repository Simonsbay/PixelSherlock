"""Debug script to compare curl vs httpx behavior."""
import json
import requests

def test_requests():
    base_url = 'http://localhost:8000/v1'
    model = '/gemini-3/space/data/baiz/checkpoints/Qwen3-VL-30B-SFT-364-lr2e5-2ep/v19-20260309-175802/checkpoint-91'

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "curl/7.81.0",
        "Accept": "*/*",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "stream": False,
    }

    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        print("\nSending request with requests...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"Status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response body: {response.text[:500]}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success!")
            print(f"Content: {result['choices'][0]['message']['content']}")
        else:
            print(f"\n❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"\n❌ Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_requests()
