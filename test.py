import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper
#from transformers import pipeline

DURATION = 10  # seconds
SAMPLE_RATE = 16000  # 16 kHz
FILENAME = "recording.wav"

def record_audio(filename, duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(filename, sample_rate, audio)
    print(f"Recording saved as {filename}")

model = whisper.load_model("base")


def main():
    record_audio(FILENAME, DURATION, SAMPLE_RATE)
    print("Transcribing...")
    result = model.transcribe(FILENAME)
    english_text = result["text"]
    print(f"Transcription: {english_text}")
    with open("transcription.txt", "w") as f:
        f.write(english_text)

if __name__ == "__main__":
    main()
