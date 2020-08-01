from utils import filesystem
from utils import sound 

def main():
    print('Initializing dta preparation...')
    audio_directories = filesystem.train_audio_files
    print(audio_directories)

if __name__ == '__main__':
    main()
