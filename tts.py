from gtts import gTTS

def text_to_speech(text, filename="output.wav"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)  # gTTS already gives usable audio
    return filename