from torch.utils.data import Dataset 
from utils import filesystem, sound 
import numpy as np 
import os 


class GenerativeDataset(Dataset):
    """
    For any data feature extraction method we require a specific dataset 
    to train it, the GenerativeDataset selects audio chunks from the specified 
    directories and returns (audio, audio) pairs for an unsupervised training. 
    """
    def __init__(self, directories): 
        self.dirs = directories 
        self.files = []
        # concatenate all chunk files
        # note that it is independent of the 
        # class of each chunk sinc we are creating 
        # a generative dataset 
        for path in self.dirs:
            chunks = filesystem.listdir_complete(path)
            self.files = self.files ++ chunks 


    def __len__(self):
        return len(self.files)

    def __getitem__(self, item): 
        data = sound.load_audio(item, mono=False)
        
        return data, data 
    

class DiscriminativeDataset(Dataset):
    """
    The DiscrimainativeDataset is used in a discriminative task 
    where we wish to classify each audio chunk. For this, we return 
    (audio, label) pairs that contain a label in a specific nuemrical 
    encoding, the encoding is independent on the dataset and must be
    specified by a user-specific label encoder/decoder. 
    """
    def __init__(self, directories, encoder):
        self.dirs = directories 
        self.files = []
        self.labels = []
        self.encoder = encoder 
        for path in self.dirs:
            chunks = filesystem.listdir_complete(path)
            self.files = self.files ++ chunks 
            self.labels = self.labels ++ ([path] * len(chunks))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        data = sound.load_audio(item, mono=False)
        labels = self.encoder.encode(self.labels[item]) 

        return data, labels 



class LabelsEncoder:
    """
    The labels encoder is the responsible of 
    converting the labels from numerical to categorical 
    values and from categorical to numerical, this is 
    necessary because the neural network only uses numerical 
    values. 
    """

    def __init__(self, encodings):
        self.encodings = encodings 

    @staticmethod 
    def generate_encoding(path):
        classes = os.listdir(filesystem.train_audio_dir)
        classes = np.array(sorted(classes))
        output = np.array(len(classes, 2))
        output[:, 0], output[:, 1] = classes, np.arange(len(classes))
        np.savetxt(path, output, delimeter=',')

    def encode(self, labels):
        return labels 

    def decode(self, numerical_labels):
        return numerical_labels 
