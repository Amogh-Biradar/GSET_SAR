import numpy as np
from scipy.io import wavfile
from scipy.signal import iirnotch, butter, filtfilt

def drone_noise_bandstopfilter(audio_recording_path):

    #1 reasing .wav file
    fs, data = wavfile.read(audio_recording_path)  #fs = sampling rate
    data = data.astype(float)  #make it float so that it can be filtered

    if(len(data) > 1):
        data = data.mean(axis=1)

    # 2. This is to filter out frequencies very close to 130 Hz
    f0 = 130.0  
    Q = 30.0    # Bigger Q = more narrower
    b_notch, a_notch = iirnotch(f0, Q, fs)  
    # these coefficients are used in the difference equation to determine what frequencies to filter

    # 3. This is to filter out rest of drone frequencies + overtones/resonant frequencies
    low_cut = 150.0  
    high_cut = 6000.0 
    order = 4         
    b_bandstop, a_bandstop = butter(order, [low_cut, high_cut], btype='bandstop', fs=fs)
    # coefficients used for same thing as in the notch.


    #this is first applying narrow filter (notch) to 130 hz
    filtered_data = filtfilt(b_notch, a_notch, data)
    # This is the 150-6kHz filter
    filtered_data = filtfilt(b_bandstop, a_bandstop, filtered_data)


    # saves filtered audio to new location
    wavfile.write('drone_filtered_output.wav', fs, filtered_data.astype(np.int16))


