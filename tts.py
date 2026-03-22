# from gtts import gTTS

# def text_to_speech(text, filename="output.wav"):
#     tts = gTTS(text=text, lang="en")
#     tts.save(filename)  # gTTS already gives usable audio
#     return filename


#The above is google text to speech and the below is ELeven Labs customized voice


from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import os

# Load environment variables 
load_dotenv()

# Initialize client
client = ElevenLabs(
    api_key="sk_57c63ab77ebd88aa0b45b9cc7111aa30e196dde4ee12e9ac",
)

def text_to_speech(text, filename="output.mp3"):

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="oGIr8duUtinux4nPetuO",  # ✅ your voice ID
        model_id="eleven_v3",
        output_format="mp3_44100_128",
    )

    # Save audio file
    with open(filename, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return filename