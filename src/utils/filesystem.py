import os
import torch
import torchaudio

# basic utility functions
def listdir_complete(path):
    """
    Equivalent to os.listdir, appends
    the complete path to each file
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


def save_chunks(chunks, class_name, sample_rate):
    """
    Given a list of chunks and a class name
    save each chunk in a separate file
    """
    dir = os.path.join(train_audio_chunks_dir, class_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i, chunk in enumerate(chunks):
        torchaudio.save(os.path.join(dir, 'chunk_' + str(i) + '.mp3'), chunk, sample_rate)


# general path variables
# each path variable is defined by *_dir
data_dir = '../data'
train_audio_dir = os.path.join(data_dir, 'train_audio')
train_audio_chunks_dir = os.path.join(data_dir, 'train_audio_chunks')
example_test_audio_dir = os.path.join(data_dir, 'example_test_audio')
# general file variables
# each file variable is defined by *_file
sample_submission_file = os.path.join(data_dir, 'sample_submission.csv')
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')
example_test_audio_metadata_file = os.path.join(data_dir, 'example_test_audio_metadata.csv')
example_test_audio_summary_file = os.path.join(data_dir, 'example_test_audio_summary.csv')
train_audio_files = listdir_complete(train_audio_dir)
example_audio_files = listdir_complete(example_test_audio_dir)
