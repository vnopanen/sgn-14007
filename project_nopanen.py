import librosa as lb
import numpy as np
from librosa.display import specshow
from matplotlib import pyplot as plt
from separate import separate

# Load audio file
file = 'police03short.wav'
audioIn, fs = lb.core.load(file)
start = fs*0
stop = fs*14 # select part of the audio from start to stop in seconds (fs*seconds)

# Call separation function to get harmonic and percussive signals and spectrograms for visualization
harmonic, percussive, specgrams = separate(audioIn[start:stop], k_max=10)


# Show spectrograms
specshow(20*np.log10(specgrams[0]), x_axis='time',y_axis='linear')
plt.title('Range-compressed spectrogram (W)')
plt.show()
plt.figure()
specshow(20*np.log10(specgrams[1]), x_axis='time', y_axis='linear')
plt.title('Harmonic spectrogram (H)')
plt.show()
plt.figure()
specshow(20*np.log10(specgrams[2]), x_axis='time',y_axis='linear')
plt.title('Percussive spectrogram (P)')
plt.show()


# Write components to audio file
file_h = file[:len(file)-4] + "_h.wav"
file_p = file[:len(file)-4] + "_p.wav"
lb.output.write_wav(file_h, harmonic, sr=fs, norm=False)
lb.output.write_wav(file_p, percussive, sr=fs, norm=False)


