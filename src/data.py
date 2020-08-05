from torch.utils.data import Dataset 
from utils import filesystem, sound 


class GenerativeDataset(Dataset):

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

    def encode(self, labels):
        return labels 

    def decode(self, numerical_labels):
        return numerical_labels 
