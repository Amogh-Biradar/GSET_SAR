# Initialize diffs as a 2D array
    diffs = [[[] for _ in range(10)] for _ in range(10)]
    
    for a, offset1 in enumerate([i / 10 for i in range(10)]):
        for b, offset2 in enumerate([j / 10 for j in range(10)]):

            l = []
            for _ in range(100):
                mics = [Mic((0.0, 0.0), 192000), Mic((0.05, 0.0, offset1), 48000), Mic((0.025, 0.0433, offset2), 48000)]
                env = getRandomEnv(mics, 100)
                for i, mic in enumerate(env.getMics()):
                    if not mic.pos == mics[0].pos:
                        l.append(abs(getTrueTDOA(mics[0], mic, env.getWave()) - getEstTDOA(mics[0], mic, env.getWave())))

            diffs[a][b].append(np.mean(l))
    
    # Find minimum value and its offset values
    min_value = float('inf')
    min_offset1 = 0
    min_offset2 = 0
    
    for a in range(10):
        for b in range(10):
            if diffs[a][b]:  # Check if list is not empty
                current_min = min(diffs[a][b])
                if current_min < min_value:
                    min_value = current_min
                    min_offset1 = a / 10
                    min_offset2 = b / 10
    
    print(f"Minimum error value: {min_value}")
    print(f"Best offset1: {min_offset1}")
    print(f"Best offset2: {min_offset2}")

