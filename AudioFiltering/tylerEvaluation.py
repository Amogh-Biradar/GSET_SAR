# base filter code is in the filterModel.py file
# write brief evaluation program to find % accuracy or F1 score

def audio_clean_MSE(true_signal, cleaned_signal):
    if len(true_signal) != len(cleaned_signal):
        raise ValueError("Signal and noise must have the same length.")
    
    return (1 / len(true_signal)) * sum((true_signal[i] - cleaned_signal[i])**2 for i in range(len(true_signal)))

def audio_clean_accuracy(true_signal, cleaned_signal):
    power = (1 / len(true_signal)) * sum(signal**2 for signal in true_signal)
    return 1 - audio_clean_MSE(true_signal, cleaned_signal) / power