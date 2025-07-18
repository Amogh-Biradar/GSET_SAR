import random, math
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import sys
import os
import matplotlib.pyplot as plt

class Mic:
    def __init__(self, pos, freq, offset_frac=0):
        self.pos = pos
        self.freq = freq
        self.offset = offset_frac / freq
    
    def getTimeReached(self, wave):
        t = self.offset
        while t * Wave.speed < math.sqrt((self.pos[0] - wave.pos[0])**2 + (self.pos[1] - wave.pos[1])**2):
            t += 1 / self.freq
        return t - 1 / self.freq, t
    
    def calcTrueTime(self, wave):
        return math.sqrt((self.pos[0] - wave.pos[0])**2 + (self.pos[1] - wave.pos[1])**2) / Wave.speed

class Environment:
    temp = 68

    def __init__(self, mics, wave):
        self.mics = mics
        self.wave = wave
    
    def getMics(self):
        return self.mics
    
    def getWave(self):
        return self.wave
    
class Wave:
    speed = 331 + 0.6 * (5/9 * (Environment.temp - 32))

    def __init__(self, pos):
        self.pos = pos

    def getDist(self):
        return math.sqrt(self.pos[0]**2 + self.pos[1]**2)

    def getPos(self):
        return self.pos


def getRandomEnv(mics, maxRad):
    center_x = mics[0].pos[0]
    center_y = mics[0].pos[1]

    r = maxRad * math.sqrt(random.random())
    theta = 2 * math.pi * random.random()
    wave = Wave((center_x + r * math.cos(theta), center_y + r * math.sin(theta)))
    return Environment(mics, wave)

def getEnvRad(mics, rad):
    center_x = mics[0].pos[0]
    center_y = mics[0].pos[1]

    theta = 2 * math.pi * random.random()
    wave = Wave((center_x + rad * math.cos(theta), center_y + rad * math.sin(theta)))
    return Environment(mics, wave)

def getEstTDOA(base_mic, other_mic, wave):
    lower_b, upper_b = base_mic.getTimeReached(wave)
    lower_o, upper_o = other_mic.getTimeReached(wave)
    return lower_o - lower_b, upper_o - upper_b

def getTrueTDOA(base_mic, other_mic, wave):
    return other_mic.calcTrueTime(wave) - base_mic.calcTrueTime(wave)

def getEstAzimuth(TDOAs, mic_positions):
    # mic_positions: list of microphone positions [(x1, y1), (x2, y2), (x3, y3)]
    # tdoas: list of TDOAs relative to mic 1 [Δt_21, Δt_31]
    v_sound = 331 + 0.6 * (5/9 * (Environment.temp - 32)) # in m/s

    ref = mic_positions[0]
    diffs = [np.array(m) - ref for m in mic_positions[1:]]
    dists = np.array([v_sound * t for t in TDOAs])  # distance differences

    A = np.vstack(diffs)
    v, _, _, _ = np.linalg.lstsq(A, dists, rcond=None)

    # Normalize direction vector
    v /= np.linalg.norm(v)
    azimuth = np.arctan2(v[1], v[0])
    return np.degrees(azimuth)  # in degrees

def posError(pos, d_2diff1, d_3diff1, mic_positions):
    """Calculate triangulation error for given position"""
    x, y = pos
    
    # Calculate distances from source to each microphone
    dist1 = math.sqrt((x - mic_positions[0][0])**2 + (y - mic_positions[0][1])**2)
    dist2 = math.sqrt((x - mic_positions[1][0])**2 + (y - mic_positions[1][1])**2)
    dist3 = math.sqrt((x - mic_positions[2][0])**2 + (y - mic_positions[2][1])**2)
    
    # Calculate predicted distance differences
    pred_d_2diff1 = dist2 - dist1
    pred_d_3diff1 = dist3 - dist1
    
    # Calculate error (sum of squared differences)
    error = (pred_d_2diff1 - d_2diff1)**2 + (pred_d_3diff1 - d_3diff1)**2
    
    return error

def triangulateSim(TDOAs, mic_positions):
    """
    Robust triangulation with multiple optimization strategies
    Mic positions gives a list of lists with the coordinates of each microphone in meters from the center of the drone
    TDOAs should be in seconds (typically very small values like 1e-4)
    """
    v_sound = 331 + 0.6 * (5/9 * (Environment.temp - 32)) # in m/s

    tdoa_2diff1 = TDOAs[0]
    tdoa_3diff1 = TDOAs[1]

    d_2diff1 = tdoa_2diff1 * v_sound
    d_3diff1 = tdoa_3diff1 * v_sound
    
    # Strategy 1: Multiple initial guesses with Nelder-Mead
    initial_guesses = [
        [0, 0],           # Center
        [1, 0], [-1, 0],  # Left/right
        [0, 1], [0, -1],  # Up/down
        [1, 1], [-1, -1], [1, -1], [-1, 1],  # Diagonals
        [0.1, 0.1], [0.5, 0.5], [2, 2]  # Various distances
    ]
    
    best_result = None
    best_error = float('inf')
    
    for guess in initial_guesses:
        try:
            result = minimize(
                posError,
                x0=guess,
                args=(d_2diff1, d_3diff1, mic_positions),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
                
        except Exception:
            continue
    
    # Strategy 2: Try L-BFGS-B with bounds
    try:
        bounds = [(-10, 10), (-10, 10)]  # Reasonable bounds for position
        for guess in initial_guesses[:5]:  # Try fewer guesses for bounded method
            result = minimize(
                posError,
                x0=guess,
                args=(d_2diff1, d_3diff1, mic_positions),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
    except Exception:
        pass
    
    # Strategy 3: Differential Evolution (global optimizer)
    try:
        bounds = [(-10, 10), (-10, 10)]
        result = differential_evolution(
            posError,
            bounds,
            args=(d_2diff1, d_3diff1, mic_positions),
            maxiter=300,
            popsize=15,
            tol=1e-10,
            seed=42
        )
        
        if result.success and result.fun < best_error:
            best_result = result
            best_error = result.fun
    except Exception:
        pass
    
    # Check if we found a good solution
    if best_result is None:
        raise ValueError("All optimization methods failed")
    
    x, y = best_result.x
    x = float(x)
    y = float(y)

    return (x, y)

def getEstDist(TDOAs, azimuth_deg, mic_positions):
    v_sound = 331 + 0.6 * (5 / 9 * (Environment.temp - 32))  # Speed of sound in m/s
    dBA = v_sound * TDOAs[0]
    dCA = v_sound * TDOAs[1]

    micA = np.array(mic_positions[0])
    micB = np.array(mic_positions[1])
    micC = np.array(mic_positions[2])

    theta = np.radians(azimuth_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])  # Unit vector from A to source

    def error_fn(r):
        source_pos = micA + r * direction
        e1 = np.linalg.norm(source_pos - micB) - np.linalg.norm(source_pos - micA) - dBA
        e2 = np.linalg.norm(source_pos - micC) - np.linalg.norm(source_pos - micA) - dCA
        return e1**2 + e2**2

    result = minimize_scalar(error_fn, bounds=(0, 1000), method='bounded', options={'xatol': 1e-6})

    return result.x

# def getInstructions(audio_files):
#     mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]

#     fs_a, sig_a = audio_files[0]
#     fs_b, sig_b = audio_files[1]
#     fs_c, sig_c = audio_files[2]

#     tdoa_ab = gcc_phat(sig_a, sig_b, fs_a)[0]
#     tdoa_ac = gcc_phat(sig_a, sig_c, fs_a)[0]

#     TDOAs = [tdoa_ab, tdoa_ac]
#     heading = getEstAzimuth(TDOAs, mic_positions)
#     meters = getEstDist(TDOAs, heading, mic_positions)
#     alertBase(f"Human found {meters} meters away at {heading} degrees.")



if __name__ == "__main__":
    mics = [Mic((0, 0), 10000), Mic((0.05, 0), 10000), Mic((0.025, 0.0433), 10000)]
    mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]

    azs = []
    EPSILON = 0.01
    
    for m in range(360):
        env = Environment(mics, Wave((100 * np.cos(np.radians(m)) * np.sqrt(np.random.uniform(0, 1)), 100 * np.sin(np.radians(m)) * np.sqrt(np.random.uniform(0, 1)))))
        x, y = triangulateSim([getEstTDOA(mics[0], mics[1], env.getWave())[1], getEstTDOA(mics[0], mics[2], env.getWave())[1]], mic_positions)
        azimuth = getEstAzimuth([getEstTDOA(mics[0], mics[1], env.getWave())[1], getEstTDOA(mics[0], mics[2], env.getWave())[1]], mic_positions)
        
        print(f"Azimuth for {m} degrees: {azimuth:.2f} degrees")
        azs.append(azimuth)
    
    unique_azs = np.unique(np.round(azs, 2))  # Round to 2 decimal places
    print("Unique azimuths:", unique_azs)   
        