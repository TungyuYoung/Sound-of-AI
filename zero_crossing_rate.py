import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd

# load audio files
wav_file = "your audio file .wav"
wav, _ = librosa.load(wav_file) # the first output is video data, the second one is sampling rate

# extract RMSE with librosa
FRAME_LENGTH = 1024
HOP_LENGTH = 512

ZCR_wav = librosa.feature.zero_crossing_rate(wav, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
frames = range(len(ZCR_wav))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

# visualize the the ZCR
plt.figure(figsize=(50, 25))
plt.plot(t, ZCR_wav, color='r')

plt.ylim(0, 0.1)
plt.title("wav_ZCR")
plt.show()