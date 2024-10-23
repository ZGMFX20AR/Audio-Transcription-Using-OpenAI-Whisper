import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper
import time
from datetime import timedelta

# Constants
DURATION = 30  # seconds
SAMPLE_RATE = 16000  # 16 kHz
FILENAME = "recording.wav"

# Function to display countdown
def countdown(seconds):
    print("Get ready to record...")
    for i in range(seconds, 0, -1):
        print(i, "...")
        time.sleep(1)
    print("Recording started!")

# Function to display live recording duration
def display_duration(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = int(time.time() - start_time)
        print(f"Recording... {elapsed}/{duration} seconds", end="\r")  # Overwrite the same line
        time.sleep(1)
    print("\nRecording finished!")  # Move to the next line after recording

# Function to record audio with live timer display
def record_audio(filename, duration, sample_rate):
    countdown(3)  # Optional countdown
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    
    # Display live recording duration
    display_duration(duration)

    sd.wait()  # Wait until recording is complete
    wavfile.write(filename, sample_rate, audio)
    print(f"Recording saved as {filename}")

# Initialize Whisper model
model = whisper.load_model("base")

# Function to save transcription with timestamps
def save_transcription_with_timestamps(transcription, output_file="transcription.txt"):
    with open(output_file, "w") as f:
        for segment in transcription["segments"]:
            start = str(timedelta(seconds=segment["start"]))
            end = str(timedelta(seconds=segment["end"]))
            text = segment["text"]
            f.write(f"[{start} --> {end}] {text}\n")

# Main function
def main():
    # Record audio with live duration display
    record_audio(FILENAME, DURATION, SAMPLE_RATE)

    # Transcribe audio
    print("Transcribing...")
    result = model.transcribe(FILENAME)

    # Display transcription in console
    english_text = result["text"]
    print(f"Full Transcription: {english_text}")

    # Save transcription with timestamps
    print("Saving transcription with timestamps...")
    save_transcription_with_timestamps(result)
    print("Transcription saved as 'transcription.txt'.")

if __name__ == "__main__":
    main()
