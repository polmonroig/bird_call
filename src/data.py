from torch.utils.data import Dataset
from utils import filesystem, sound
import csv
import os


class GenerativeDataset(Dataset):
    """
    For any data feature extraction method we require a specific dataset
    to train it, the GenerativeDataset selects audio chunks from the specified
    directories and returns (audio, audio) pairs for an unsupervised training.
    """
    def __init__(self, chunks, transforms=None):
        self.files = chunks
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        data = sound.load_audio(self.files[item], mono=False)
        data = data[0][0]
        if self.transforms:
            data = self.transforms(data)

        return data, data 


class DiscriminativeDataset(Dataset):
    """
    The DiscrimainativeDataset is used in a discriminative task
    where we wish to classify each audio chunk. For this, we return
    (audio, label) pairs that contain a label in a specific nuemrical
    encoding, the encoding is independent on the dataset and must be
    specified by a user-specific label encoder/decoder.
    """
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
        self.encoder = encoder

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
        self.encoder = {}
        self.decoder = {}
        for code in encodings:
            self.encoder[code[0]] = code[1]
            self.decoder[code[1]] = code[0]

    @staticmethod
    def generate_encoding(path):
        labels = os.listdir(filesystem.train_audio_dir)
        labels = sorted(labels)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i, label in enumerate(labels):
                writer.writerow([label, i])



    def encode(self, labels):
        return self.encoder[labels]

    def decode(self, numerical_labels):
        return self.decoder[numerical_labels]
