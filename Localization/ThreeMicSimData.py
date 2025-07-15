import numpy as np
import matplotlib.pyplot as plt
from audioSim import Mic, Source, Environment, Wave, getRandomEnv, getTrueTDOA, getEstTDOA
from triangulateSim import triangulateSim

def get3micSimData(n):
        mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]
        maxRad = 100

        data = [] # structure:[[[tdoas], (source_pos)], ...]
        for _ in range(n):
                mics = [Mic((0.0, 0.0)), Mic((0.05, 0.0)), Mic((0.025, 0.0433))]
                env = getRandomEnv(mics, maxRad)
                env.runSim()
                true_tdoas = [getTrueTDOA(mics[0], mics[1], env.getWave()), getTrueTDOA(mics[0], mics[2], env.getWave())]
                measured_tdoas = [getEstTDOA(mics[0], mics[1]), getEstTDOA(mics[0], mics[2])]
                data.append((measured_tdoas, true_tdoas))
        print(len(data), "data points generated")
        measured, true = zip(*[sim for sim in data])
        return measured, true

def determined3micSimData(n):
        mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]
        maxRad = 100

        data = []
        for deg in range(0, 359):
                mics = [Mic((0.0, 0.0)), Mic((0.05, 0.0)), Mic((0.025, 0.0433))]
                env = Environment(mics, Wave((maxRad * np.cos(np.radians(deg)), maxRad * np.sin(np.radians(deg)))))
                env.runSim()
                true_tdoas = [getTrueTDOA(mics[0], mics[1], env.getWave()), getTrueTDOA(mics[0], mics[2], env.getWave())]
                measured_tdoas = [getEstTDOA(mics[0], mics[1]), getEstTDOA(mics[0], mics[2])]
                data.append((measured_tdoas, true_tdoas))
        print(len(data), "data points generated")
        measured, true = zip(*[sim for sim in data])
        return measured, true


