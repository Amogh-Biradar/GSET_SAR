import numpy as np
import csv
from audioSim import Mic, getRandomEnv, getTrueTDOA, getEstTDOA

if __name__ == "__main__":
    mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]
    maxRad = 100

    data = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            offset1 = i / 100  # Convert index to offset value
            offset2 = j / 100  # Convert index to offset value
            
            mics = [Mic((0.0, 0.0), 192000), Mic((0.05, 0.0), 48000, offset1), Mic((0.025, 0.0433), 48000, offset2)]
            env = getRandomEnv(mics, maxRad)
            tdoadiff1 = getEstTDOA(mics[0], mics[1], env.getWave()) - getTrueTDOA(mics[0], mics[1], env.getWave())
            tdoadiff2 = getEstTDOA(mics[0], mics[2], env.getWave()) - getTrueTDOA(mics[0], mics[2], env.getWave())
            data[i, j] = (abs(tdoadiff1) + abs(tdoadiff2)) / 2
    
    with open('triangulation_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
    print("Data saved to triangulation_data.csv")
