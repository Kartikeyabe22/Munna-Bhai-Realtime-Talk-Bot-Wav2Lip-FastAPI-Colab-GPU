import uuid
import os
import requests

API_URL = "https://munna.instatunnel.my/generate"

def generate_video(audio_path):
    # ✅ unique filename
    unique_id = str(uuid.uuid4())
    output_path = f"videos/{unique_id}.mp4"

    os.makedirs("videos", exist_ok=True)

    with open(audio_path, "rb") as f:
        files = {"audio": f}
        response = requests.post(API_URL, files=files)

    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text[:1000])

    if response.status_code != 200:
        raise Exception(f"Video generation failed: {response.status_code}")

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path