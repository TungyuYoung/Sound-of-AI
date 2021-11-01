import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd

# load audio files
wav_file = "your audio file .wav"
#ipd.Audio(wav_file)
wav, sr = librosa.load(wav_file)# 第一个输出是video data,第二个是sampling rate
print(wav.shape, type(wav.shape)) #
sample_duration = 1/sr
print(f"Duration of 1 sample is: {sample_duration:.6f} seconds") # duration of 1 sample

sigal_duration = sample_duration * len(wav)
print(f"Duration of 1 sample is: {sigal_duration:.2f} seconds")

#visualize the waveform
# plt.figure(figsize=(15, 5))
# librosa.display.waveplot(wav, alpha=0.5)
# plt.title('wav')
# plt.ylim(-1, 1)
# plt.show()

# calculate the amplitude envelop
FRAME_SIZE = 1024
HOP_LENGTH = 512 # 步长

def amplitude_envelope_no(signal, frame_size): #non-overlapping
    amplitude_envelope = []
    # calculate AE of each frame
    for i in range(0, len(signal), frame_size):
        current_frame_amplitude_envelope = max(signal[i:i+frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)

    return np.array(amplitude_envelope)

def amplitude_envelope_o(signal, frame_size, hop_length): #non-overlapping
    amplitude_envelope = []
    # calculate AE of each frame
    for i in range(0, len(signal), hop_length):
        current_frame_amplitude_envelope = max(signal[i:i+frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)

    return np.array(amplitude_envelope)

def fancy_AE_o(signal, frame_size, hop_length):
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])


AE_wav_f = fancy_AE_o(wav, FRAME_SIZE, HOP_LENGTH)
AE_wav = amplitude_envelope_o(wav, FRAME_SIZE, HOP_LENGTH)

frames = range(0, AE_wav_f.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

# visualize amplitude envelope for audio files
plt.figure(figsize=(15, 5))
librosa.display.waveplot(wav, alpha=0.5)
plt.plot(t, AE_wav_f, color = 'r')
plt.title('wav')
plt.ylim(-1, 1)
plt.show()




