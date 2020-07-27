import os

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
