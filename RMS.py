import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd

# RMSE myself
def RMSE(signal, frame_length, hop_length):
    rmse = []
    for i in range(0, len(signal), hop_length):
        RMSE_current_frame = np.sqrt(sum(signal[i:i+frame_length]**2)/frame_length)
        rmse.append(RMSE_current_frame)
    return np.array(rmse)

# load audio files
wav_file = "your audio file .wav"
# ipd.Audio(wav_file)
wav, _ = librosa.load(wav_file) # the first output is video data, the second one is sampling rate

# extract RMSE with librosa
FRAME_LENGTH = 1024
HOP_LENGTH = 512

RMS_wav = RMSE(wav, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
# RMS_wav = librosa.feature.rms(wav, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH) # use librosa

# plot the RMSE for all the music pieces

frames = range(len(RMS_wav))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

plt.figure(figsize=(15, 5))
librosa.display.waveplot(wav, alpha=0.5)
plt.plot(t, RMS_wav, color='r')
plt.ylim((-1, 1))
plt.title('wav_RMSE')
plt.show()
