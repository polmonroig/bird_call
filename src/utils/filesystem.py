import os

# basic utility functions 
def listdir_complete(path):
    """
    Equivalent to os.listdir, appends 
    the complete path to each file 
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

 

# general path variables
# each path variable is defined by *_dir 
data_dir = '../data'
train_audio_dir = os.path.join(data_dir, 'train_audio')
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
