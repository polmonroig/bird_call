import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    """
    The FeatureExtractor is an autoencoder model that works by learning
    a mapping of the raw audio wave into a smaller dimensional space
    with high level features. This model is train in an unsupervised manner
    as an undercomplete autoencoder. The objective is to create a model
    that extracts the most important features of the audio, similarly to
    a log-mel audio transform but with a noise reduction. It must learn to
    represent audio without the noise.

    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=16,
                            kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm1d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=3, padding=0),
                        nn.Conv1d(in_channels=16, out_channels=32,
                            kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=3, padding=0),
                        nn.Conv1d(in_channels=32, out_channels=64,
                            kernel_size=3, stride=1, padding=0),
                        nn.ReLU(inplace=True),
                        )
        self.decoder = nn.Sequential(
                        nn.Conv1d(in_channels=64, out_channels=32,
                            kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=3, padding=0),
                        nn.Conv1d(in_channels=32, out_channels=16,
                            kernel_size=3, stride=1,  padding=0),
                        nn.BatchNorm1d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=3, padding=0),
                        nn.Conv1d(in_channels=16, out_channels=1,
                            kernel_size=3, stride=1, padding=0)
                        )


    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))



class Classifier(nn.Module):
    """
    The Classifier is a small neural netork that learns to classify
    bird audio waves based on the features that are extracted previosly.
    An aribrary encoder must be provide that is compatible with the
    classification layers.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder



def train_step(model, data_loader, optimizer, loss_criterion, verbose_epochs, device):
    model.train()
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        out = model(data)
        print('Out shape:', out.shape)
        print('Data shape:', data.shape)
        loss = loss_criterion(out, data)
        loss.backward()
        optimizer.step()
        if i % verbose_epochs == 0:
            print('Loss:', loss)



def eval_step(model, data_loader, loss_criterion, verbose_epochs, device):
    model.eval()
    for i, data in enumerate(data_loader):
        data, labels = data
        data = data.to(device)
        out = model(data)
        loss = loss_criterion(out, data)
        if i % verbose_epochs == 0:
            print('Loss:', loss)
