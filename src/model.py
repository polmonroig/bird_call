import torch.nn as nn
import torch
import wandb


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
        self.encoder = nn.ModuleList(
                    [
                        nn.Conv1d(in_channels=1, out_channels=16,
                            kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=3, padding=0, return_indices=True),
                        nn.Conv1d(in_channels=16, out_channels=32,
                            kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=3, padding=0, return_indices=True),
                        nn.Conv1d(in_channels=32, out_channels=64,
                            kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True)
                        ]
                        )
        self.decoder = nn.ModuleList([
                        nn.Conv1d(in_channels=64, out_channels=32,
                            kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxUnpool1d(kernel_size=3, padding=0),
                        nn.Conv1d(in_channels=32, out_channels=16,
                            kernel_size=3, stride=1,  padding=1),
                        nn.BatchNorm1d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxUnpool1d(kernel_size=3, padding=0),
                        nn.Conv1d(in_channels=16, out_channels=1,
                            kernel_size=3, stride=1, padding=1)
                            ]
                        )

        self.indices = []


    def encode(self, x):
        self.indices = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool1d):
                x, ind = layer(x)
                self.indices.append(ind)
            else:
                x = layer(x)

        return x

    def decode(self, x):
        index = len(self.indices) - 1
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool1d):
                x = layer(x, self.indices[index])
                index -= 1
            else:
                x = layer(x)
        return x

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


    def forward(self, x):
        x = self.encoder.encode(x) # feature extraction  
        for layer in self.layers:
            x = layer(x)
        return x



def train_step(model, data_loader, optimizer, loss_criterion, verbose_epochs, device):
    model.train()
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        print(data.shape)
        out = model(data)

        loss = loss_criterion(out, data)
        loss.backward()
        optimizer.step()
        if i % verbose_epochs == 0:
            print('Train Loss:', loss.item())
            wandb.log({'Train Loss' : loss.item()})



def eval_step(model, data_loader, loss_criterion, verbose_epochs, device):
    model.eval()
    for i, data in enumerate(data_loader):
        data, labels = data
        data = data.to(device).reshape(data.shape[0], 1, -1)
        out = model(data)
        loss = loss_criterion(out, data)
        if i % verbose_epochs == 0:
            print('Eval Loss:', loss.item())
            wandb.log({'Eval Loss' : loss.item()})
