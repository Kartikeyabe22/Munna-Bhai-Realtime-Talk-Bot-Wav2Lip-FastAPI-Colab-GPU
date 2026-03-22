from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import shutil
import uuid
import subprocess
import uvicorn
import threading
import os

app = FastAPI()

# 🔥 CONSTANT PATHS (IMPORTANT)
BASE_DIR = "/content/Wav2Lip"
MODEL_PATH = "/content/Wav2Lip/checkpoints/wav2lip_gan.pth"
FACE_PATH = "/content/Wav2Lip/munna.png"

@app.post("/generate")
async def generate_video(audio: UploadFile = File(...)):
    try:
        # 🔥 Ensure correct working directory
        os.chdir(BASE_DIR)

        # 🔥 Save input audio
        audio_filename = f"input_{uuid.uuid4()}.wav"
        with open(audio_filename, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        output_file = f"output_{uuid.uuid4()}.mp4"

        # 🔥 Check required files
        if not os.path.exists(MODEL_PATH):
            return JSONResponse({"error": "Model not found"}, status_code=500)

        if not os.path.exists(FACE_PATH):
            return JSONResponse({"error": "Face image not found"}, status_code=500)

        # 🔥 COMMAND (FINAL FIXED)
        command = f"""
        python inference.py \
        --checkpoint_path {MODEL_PATH} \
        --face {FACE_PATH} \
        --audio {audio_filename} \
        --outfile {output_file}
        """

        print("🚀 Running Wav2Lip...")
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )

        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

        # 🔥 Check if output created
        if not os.path.exists(output_file):
            return JSONResponse({
                "error": "Video not generated",
                "stderr": result.stderr
            }, status_code=500)

        print("✅ Video generated:", output_file)

        return FileResponse(output_file, media_type="video/mp4")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# 🔥 MAIN ENTRY
if __name__ == "__main__":

    # 🔥 Start FastAPI
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    thread = threading.Thread(target=run_api)
    thread.start()
    
#Cloudflare tunnel does not provide a stable url so we are using instatunnel which is stable url


    # # 🔥 Install cloudflared (safe)
    # os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared")
    # os.system("chmod +x cloudflared")

    # print("🔥 Starting Cloudflare tunnel...")

    # # 🔥 Start tunnel
    # os.system("./cloudflared tunnel --url http://localhost:8000")
    
    

    os.system("instatunnel tunnel 8000 --subdomain munna")