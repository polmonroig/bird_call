import torchaudio
import torch

def load_audio(path, mono=True):
    """
    Return the audio data in mono form of a file and the sample
    rate
    """
    audio = torchaudio.load(path)
    if mono:
        return (audio[0][0], audio[1])
    else:
        return audio


def divide_into_chunks(audio, chunk_size, min_size):
    """
    Given an audiio waveform, it divides it into equal size
    chunks. The final chunk may be discarded or padded.

    audio: the waveform to subdivide
    chunk_size: the size of the chunk in seconds
    min_size: defines the minimum size that the chunk must have to padd it

    """
    chunks = torch.split(audio[0], int(chunk_size * audio[1]))
    last_chunk_duration = len(chunks[-1]) / audio[1]
    if last_chunk_duration >= min_size:
        return chunks[:-1]
    else:
        return chunks
