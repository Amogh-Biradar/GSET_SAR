
from gcc_phat import gcc_phat
import numpy as np
from scipy.io import wavfile
from scipy.optimize import minimize




def to_mono(sig):
    return sig.mean(axis=1) if sig.ndim > 1 else sig



    
def posError(pos, diff, mic_positions=[[0.0,0.0], [1.0, 0.0]]):
    
    
    
    
    x, y = pos
    d1 = np.sqrt((x - mic_positions[0][0])**2 + (y - mic_positions[0][1])**2)
    d2 = np.sqrt((x - mic_positions[1][0])**2 + (y - mic_positions[1][1])**2)
    
        
    return (d2 - d1 - diff)**2
    
    
    
    


def triangulatePosition(audio, temp = 68, mic_positions=[[0.0,0.0], [1.0, 0.0]]):
   
    v_sound = 331 + 0.606 * (5/9 * (temp - 32)) # in m/s

    fs1, sig1 = audio[0]
    fs2, sig2 = audio[1]
    d_2diff1 = v_sound*gcc_phat(sig2, sig1, fs=fs1)[0]
    

    result = minimize(
        posError,
        x0=[0,0],
        args=(d_2diff1, mic_positions),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
    )

    
    return result.x
    


