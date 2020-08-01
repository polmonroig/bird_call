from utils import filesystem
from utils import sound
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Prepare data from training')
    parser.add_argument('--chunk_size', dest='chunk_size',
        help='Audio chunk size in seconds', type=integer)
    parser.add_argument('--min_chunk_size', dest='min_chunk_size', type='integer',
        help='Defines the minimum size that the chunk must have to padd it')

    return parser

def main():
    """
    Prepares the data for training, this includes:
        - Data augmentation
        - Data filtering
        - Waveform chunk subdivision
    """
    print('Initializing data preparation...')
    audio_directories = filesystem.train_audio_files
    sample_class = audio_directories[0]
    print('Selecting class', sample_class)
    audio_files = filesystem.listdir_complete(sample_class)
    print('Total audio files', len(audio_files))
    print('Loading file:', audio_files[0])
    sample_audio = sound.load_audio(audio_files[0])
    print('Waveform length:', len(sample_audio[0][0]))
    print('Waveform sample rate:', sample_audio[1])
    sample_audio = sound.divide_into_chunks(sample_audio, )

if __name__ == '__main__':
    main()
