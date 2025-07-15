import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
from audioSim import Mic, Environment, Wave, getTrueTDOA

def posError(pos, d_2diff1, d_3diff1, mic_positions):
    x, y = pos
    d1 = np.sqrt((x - mic_positions[0][0])**2 + (y - mic_positions[0][1])**2)
    d2 = np.sqrt((x - mic_positions[1][0])**2 + (y - mic_positions[1][1])**2)
    d3 = np.sqrt((x - mic_positions[2][0])**2 + (y - mic_positions[2][1])**2)
    return (d2 - d1 - d_2diff1)**2 + (d3 - d1 - d_3diff1)**2

def triangulateSim(TDOAs, mic_positions, temp=68):
    """
    Robust triangulation with multiple optimization strategies
    Mic positions gives a list of lists with the coordinates of each microphone in meters from the center of the drone
    TDOAs should be in seconds (typically very small values like 1e-4)
    temp is the temperature in degrees Fahrenheit
    """
    v_sound = 331 + 0.6 * (5/9 * (temp - 32)) # in m/s

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
    
    return (float(x), float(y))

if __name__ == "__main__":
    mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]
    tdoas = [-1e-4, 2e-4]
    x, y = triangulateSim(tdoas, mic_positions)

    mics = [Mic((0.0, 0.0)), Mic((0.05, 0.0)), Mic((0.025, 0.0433))]
    env = Environment(mics, Source((x, y)))
    mics = env.getMics()

    print("Diff: " + str(tdoas[0] - getTrueTDOA(mics[0], mics[1], env.getSource())))
    print("Diff: " + str(tdoas[1] - getTrueTDOA(mics[0], mics[2], env.getSource())))

    print()