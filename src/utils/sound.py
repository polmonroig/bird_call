import librosa

def load_audio(path, mono=True):
    """
    Return the audio data in mono form of a file  
    """
    return librosa.load(path=path, mono=mono)[0]
