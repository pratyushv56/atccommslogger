import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


fs, data = wavfile.read('test1.wav')  # fs = sampling rate


if data.ndim > 1:
    data = data[:, 0]


lowcut = 300.0
highcut = 3500.0
filtered_data = bandpass_filter(data, lowcut, highcut, fs)


from scipy.io.wavfile import write
write("filtered_output.wav", fs, filtered_data.astype(np.int16))
