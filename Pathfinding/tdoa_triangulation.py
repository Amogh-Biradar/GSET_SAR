from gcc_phat import gcc_phat
import numpy as np
from scipy.io import wavfile
from scipy.optimize import minimize

# def to_mono(sig):
#    return sig.mean(axis=1) if sig.ndim > 1 else sig
#

def posError(pos, d_2diff1, d_3diff1, mic_positions=[[0.0,0.0], [1.0, 0.0], [0.5, 0.866]]):
    x, y = pos
    d1 = np.sqrt((x - mic_positions[0][0])**2 + (y - mic_positions[0][1])**2)
    d2 = np.sqrt((x - mic_positions[1][0])**2 + (y - mic_positions[1][1])**2)
    d3 = np.sqrt((x - mic_positions[2][0])**2 + (y - mic_positions[2][1])**2)
    return (d2 - d1 - d_2diff1)**2 + (d3 - d1 - d_3diff1)**2
    

def triangulatePosition(audio, temp = 68, mic_positions=[[0.0,0.0], [1.0, 0.0], [0.5, 0.866]]):
    """
    Mic positions gives a list of lists with the coordinates of each microphone in meters from the center of the drone
    Audio is a list of references to the 3 wav files
    temp is the temperature in degrees Fahrenheit
    """
    v_sound = 331 + 0.606 * (5/9 * (temp - 32)) # in m/s

    fs1, sig1 = audio[0]
    fs2, sig2 = audio[1]
    fs3, sig3 = audio[2]

    # Maybe convert to mono if necessary

    tdoa_2diff1 = gcc_phat(sig2, sig1, fs=fs1)[0]
    tdoa_3diff1 = gcc_phat(sig3, sig1, fs=fs1)[0]

    print("TDOA 2-1:", tdoa_2diff1)
    print("TDOA 3-1:", tdoa_3diff1)

    d_2diff1 = tdoa_2diff1 * v_sound
    d_3diff1 = tdoa_3diff1 * v_sound

    result = minimize(
        posError,
        x0=[0,0],
        args=(d_2diff1, d_3diff1, mic_positions),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    return result.x

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_files = [
        wavfile.read(os.path.join(base_dir, "test_mic_a.wav")),
        wavfile.read(os.path.join(base_dir, "distorted_test_mic_a_0.2_shift.wav"))
    ]
    fs1, sig1 = audio_files[0]
    fs2, sig2 = audio_files[1]
    print(gcc_phat(sig1, sig2, fs=fs1)[0])
