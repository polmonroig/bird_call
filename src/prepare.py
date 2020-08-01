from utils import filesystem
from utils import sound
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Prepare data from training')
    parser.add_argument('--chunk_size', dest='chunk_size',
        help='Audio chunk size in seconds', type=int)
    parser.add_argument('--min_chunk_size', dest='min_chunk_size', type=float,
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
    parser = get_parser()
    args = parser.parse_args()
    chunk_size = args.chunk_size
    min_chunk_size = args.min_chunk_size


    audio_directories = filesystem.train_audio_files
    sample_class = audio_directories[0]
    print('Selecting class', sample_class)
    audio_files = filesystem.listdir_complete(sample_class)
    print('Total audio files', len(audio_files))
    print('Loading file:', audio_files[0])
    sample_audio = sound.load_audio(audio_files[0])
    print('Waveform length:', len(sample_audio[0]))
    print('Waveform sample rate:', sample_audio[1])
    sample_audio = sound.divide_into_chunks(sample_audio, chunk_size, min_chunk_size)
    print('Total chunks created:', len(sample_audio))
    print('Last chunk:', len(sample_audio[-2]))

if __name__ == '__main__':
    main()
