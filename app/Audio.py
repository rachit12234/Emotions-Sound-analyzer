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

time_history = []
rms_history = []
peak_history = []
freq_history = []
start_time = time.time()
audio_history = []

fig , axes = plt.subplots(2,2)
plt.ion()

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

    
    current_time = time.time()-start_time

    time_history.append(current_time)
    rms_history.append(current_rms)
    peak_history.append(current_peak)
    freq_history.append(current_freq)
    audio_history.append(audio_chunk)

    rolling_buffer *= 0.99  # smooth decay of old signal

    # current_rms = 0
    # current_peak = 0
    # current_freq = 0

def update_plot(frame):
    axes[0,0].cla()
    axes[0,0].set_title("RMS Over Time")
    axes[0,0].plot(time_history, rms_history, 'g-')

    axes[0,1].cla()
    axes[0,1].set_title("Peak Amplitude Over Time")
    axes[0,1].plot(time_history, peak_history, 'b-')

    axes[1,0].cla()
    axes[1,0].set_title("Dominant Frequency Over Time")
    axes[1,0].plot(time_history, freq_history, 'r-')
    axes[1,0].set_ylim(0, 500)  # 0–20 kHz range

    axes[1,1].cla()
    axes[1,1].set_title("Audio Waveform")
    if len(audio_history) > 0:
        waveform = np.concatenate(audio_history)  # flatten all chunks
        time_axis = np.linspace(0, len(waveform) / fs, len(waveform))
        axes[1,1].plot(time_axis, waveform, 'k-')
        axes[1,1].set_xlim(time_axis[-1]-5, time_axis[-1])  # show last 5 seconds

ani = FuncAnimation(fig, update_plot, interval=200)
plt.show(block=False)

with sd.InputStream(callback=audio_callback, channels=channels, samplerate=fs, blocksize=chunk):
    print("🎙️ Rolling buffer live audio stream started... Press Ctrl+C to stop.")
    try:
        while True:
            plt.pause(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")