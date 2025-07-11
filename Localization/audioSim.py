import random, math

class Mic:
    def __init__(self, pos):
        self.pos = pos
        self.time_reached = None

    def getTime(self):
        return self.time_reached
    
    def checkReached(self, wave):
        if not self.time_reached == None:
            return True
        if (self.pos[0] - wave.source_pos[0])**2 + (self.pos[1] - wave.source_pos[1])**2 <= wave.r**2:
            self.time_reached = wave.r / Wave.speed
            return True
        return False
        

class Source:
    def __init__(self, pos):
        self.pos = pos
    
    def emit(self):
        return Wave(self.pos)

class Environment:
    temp = 68
    freq = 48000

    def __init__(self, mics, source):
        self.mics = mics
        self.source = source
    
    def getMics(self):
        return self.mics
    
    def checkComplete(self):
        complete = True
        for mic in self.mics:
            if mic.getTime() == None:
                complete = False
                break
        return complete
    
    def runSim(self):
        t = 0
        w = self.source.emit()
        while self.checkComplete() == False:
            t += 1 / self.freq
            w.incRadius()
            for mic in self.mics:
                mic.checkReached(w)


class Wave:
    speed = 331 + 0.6 * (5/9 * (Environment.temp - 32))

    def __init__(self, source_pos):
        self.source_pos = source_pos
        self.r = 0
    
    def incRadius(self):
        self.r += Wave.speed / Environment.freq

def getRandomEnv(mic_positions, maxRad):
    mics = [Mic(pos) for pos in mic_positions]
    center_x = mics[0].pos[0]
    center_y = mics[0].pos[1]

    r = maxRad * math.sqrt(random.random())
    theta = 2 * math.pi * random.random()
    source = Source((center_x + r * math.cos(theta), center_y + r * math.sin(theta)))
    return Environment(mics, source)

def getTDOA(base_mic, other_mic):
    if base_mic.time_reached == None or other_mic.getTime() == None:
            raise ValueError("The wave must have reached both microphones.")
    return other_mic.getTime() - base_mic.getTime()