import sounddevice as sd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fs = 44100
chunk = 1024
duration = 5 
channels = 1
dtype = 'int16'
default={}

buffer_duration = 5  # seconds
buffer_size = fs * buffer_duration
rolling_buffer = np.zeros(buffer_size, dtype=np.float32)  # initialize buffer

def rms(a):
    rms = np.sqrt(np.mean(a**2))
    return rms

def peak(a):
    peak = np.max(np.abs(a))
    return peak

def dominant_freq(a, fs):
    a = a.flatten()
    N = len(a)
    freq_spectrum = np.fft.fft(a)
    freq_values = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(freq_spectrum[:N//2])  # only positive freqs
    freq_values = freq_values[:N//2]

    dom_freq = freq_values[np.argmax(magnitude)]
    return dom_freq


print("We need to record a sample to adjust to your normal voice \n")
key=input("Press 'R' to start recording a 5 sec sample or 'Q' to discard.\n")

if (key.lower() == 'r'):
    for i in range(3,0,-1):
        print(i)
        time.sleep(1)
    print("Recording...")
    sample = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype=dtype)
    sd.wait()
    print(sample)

    samprms = rms(sample) 
    sampeak = peak(sample)
    samfreq = dominant_freq(sample, fs)
    default["RMS"] = samprms
    default["PEAK"] = sampeak
    default["FREQ"] = samfreq
    print("Sample recorded:",default)
    print(f"RMS={samprms:.5f}, PEAK={sampeak:.5f}, FREQ={samfreq:.2f} Hz")

def audio_callback(indata, frames, time_info, status):
    global rolling_buffer
    audio_chunk = indata[:, 0].astype(np.float32)

    # shift old buffer left and append new chunk
    rolling_buffer = np.roll(rolling_buffer, -len(audio_chunk))
    rolling_buffer[-len(audio_chunk):] = audio_chunk

    # compute metrics on current buffer
    current_rms = rms(rolling_buffer)
    current_peak = peak(rolling_buffer)
    current_freq = dominant_freq(rolling_buffer, fs)

    print(f"RMS={current_rms:.5f}, PEAK={current_peak:.5f}, FREQ={current_freq:.2f} Hz")

with sd.InputStream(callback=audio_callback, channels=channels, samplerate=fs, blocksize=chunk):
    print("🎙️ Rolling buffer live audio stream started... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")