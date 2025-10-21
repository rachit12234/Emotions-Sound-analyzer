# audio_analysis.py
import numpy as np

def analyze_audio(audio_chunk, fs):
    # RMS (energy)
    rms = np.sqrt(np.mean(audio_chunk**2))
    
    # Peak amplitude
    peak = np.max(np.abs(audio_chunk))
    
    # Frequency spectrum (FFT)
    freq_values = np.fft.fft(audio_chunk)
    freq_range = np.fft.fftfreq(len(audio_chunk), 1/fs)
    
    # Get dominant frequency
    positive_freqs = freq_range[:len(freq_range)//2]
    magnitudes = np.abs(freq_values[:len(freq_values)//2])
    dominant_freq = positive_freqs[np.argmax(magnitudes)]
    
    print(f"RMS: {rms:.4f}, Peak: {peak:.4f}, Dominant Freq: {dominant_freq:.2f} Hz")