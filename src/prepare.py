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
        - Chunk selection 
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
    class_name = sample_class.split('/')[-1]
    print('Total audio files', len(audio_files))
    audio_index = 0
    for audio in audio_files: 
        print('Loading file:', audio)
        sample_audio = sound.load_audio(audio)
        sample_rate = sample_audio[1]
        print('Waveform length:', len(sample_audio[0]))
        print('Waveform sample rate:', sample_audio[1])
        sample_audio = sound.divide_into_chunks(sample_audio[0], chunk_size, min_chunk_size)
        print('Total chunks created:', len(sample_audio))
        filesystem.save_chunks(sample_audio, class_name, audio_index)
        audio_index += len(sample_audio)


if __name__ == '__main__':
    main()
