import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

fs = 44100
duration = 3

print("Recording...")
audio = sd.rec(int(duration*fs) , samplerate= fs , channels= 1)
sd.wait()
print("recording done")

array = audio.flatten()
print (array)

length = len(array)
freq_values= np.fft.fft(array)
freq_range = np.fft.fftfreq(length , 1/fs)

pos = freq_range>0
freq_values = np.abs(freq_values[pos])
freq_range = freq_range[pos]

#plotting charts
number_array=[ ]
for i in range(0,length):
        number_array.append(i)

#plt.plot(np.array(number_array)/fs,array , color="green")
plt.plot(freq_range,freq_values)
plt.show()