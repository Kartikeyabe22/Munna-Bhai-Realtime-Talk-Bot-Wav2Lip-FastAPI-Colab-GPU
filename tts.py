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
    api_key="sk_fbce4e1193f83e78dd6ea965a4c967bee708757cfd3dcaf1",
)

def text_to_speech(text, filename="output.mp3"):

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="olgAEEIlAMvn96kczoCa",  # ✅ your voice ID
        model_id="eleven_v3",
        output_format="mp3_44100_128",
    )

    # Save audio file
    with open(filename, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return filename