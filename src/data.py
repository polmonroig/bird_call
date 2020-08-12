from torch.utils.data import Dataset
from utils import filesystem, sound
import torch
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
            data = torch.from_numpy(self.transforms(data.reshape(1, -1))).float()

        return data, data


class DiscriminativeDataset(Dataset):
    """
    The DiscrimainativeDataset is used in a discriminative task
    where we wish to classify each audio chunk. For this, we return
    (audio, label) pairs that contain a label in a specific nuemrical
    encoding, the encoding is independent on the dataset and must be
    specified by a user-specific label encoder/decoder.
    """
    def __init__(self, files, labels, encoder, transforms):
        self.files = files
        self.labels = labels
        self.encoder = encoder
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        data = sound.load_audio(self.files[item], mono=False)[0][0]
        labels = self.encoder.encode(self.labels[item])
        if self.transforms:
            data = torch.from_numpy(self.transforms(data.reshape(1, -1))).float()

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
        for index, row in encodings.iterrows():
            self.encoder[row['code']] = row['id']
            self.decoder[row['id']] = row['code']

    @staticmethod
    def generate_encoding(path):
        labels = os.listdir(filesystem.train_audio_dir)
        labels = sorted(labels)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['code', 'id'])
            for i, label in enumerate(labels):
                writer.writerow([label, i])



    def encode(self, labels):
        return self.encoder[labels]

    def decode(self, numerical_labels):
        return self.decoder[numerical_labels]
