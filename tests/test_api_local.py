import requests
import json
import sys

URL = "http://localhost:8000/v2/tts/pipeline"

payload = {
    "text": "这是一个测试文本。Hello World.",
    # "text_lang": "zh",  # MUST NOT INCLUDE THIS
    "ref_audio_path": "z.refs/main.wav",
    "prompt_text": "提示文本",
    "prompt_lang": "zh",
    "speed_factor": 1.05
    # Other fields use defaults
}

print(f"Sending request to {URL}...")
try:
    resp = requests.post(URL, json=payload, timeout=30)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print("Success!")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("Error Response:")
        try:
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        except:
            print(resp.text)
except Exception as e:
    print(f"Request failed: {e}")
    sys.exit(1)
