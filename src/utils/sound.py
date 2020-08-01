import torchaudio
import torch

def load_audio(path, mono=True):
    """
    Return the audio data in mono form of a file and the sample
    rate
    """
    return torchaudio.load(path)


def divide_into_chunks(audio, chunk_size, min_size):
    """
    Given an audiio waveform, it divides it into equal size
    chunks. The final chunk may be discarded or padded.

    audio: the waveform to subdivide
    chunk_size: the size of the chunk in seconds
    min_size: defines the minimum size that the chunk must have to padd it

    """
    return torch.tensor(audio)
